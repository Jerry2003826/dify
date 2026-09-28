"""
Comprehensive unit tests for services.app_generate_service.AppGenerateService.

Covers:
  - _build_streaming_task_on_subscribe  (streams / pubsub / exception / idempotency)
  - generate                           (COMPLETION / AGENT_CHAT / CHAT / ADVANCED_CHAT / WORKFLOW / invalid mode,
                                         streaming & blocking, billing, quota-refund-on-error, rate_limit.exit,
                                         streaming slot held until close)
  - _get_max_active_requests            (all limit combos)
  - generate_single_iteration           (ADVANCED_CHAT / WORKFLOW / invalid mode)
  - generate_single_loop                (ADVANCED_CHAT / WORKFLOW / invalid mode)
  - generate_more_like_this
  - _get_workflow                       (debugger / non-debugger / specific id / invalid format / not found)
  - get_response_generator              (ended / non-ended workflow run)
"""

import gc
import json
import threading
import uuid
from collections.abc import Callable, Generator, Mapping
from typing import cast
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.orm import Session

import services.app_generate_service as ags_module
from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.features.rate_limiting.rate_limit import RateLimit, RateLimitGenerator
from enums import DeploymentEdition, QuotaType
from graphon.enums import WorkflowExecutionStatus
from models.account import Account
from models.enums import AppStatus, CreatorUserRole
from models.model import App, AppMode
from models.workflow import Workflow, WorkflowRun, WorkflowRunTriggeredFrom, WorkflowType
from services.app_generate_service import AppGenerateService
from services.errors.app import (
    TriggerWorkflowServiceModeUnavailableError,
    WorkflowIdFormatError,
    WorkflowNotFoundError,
)


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------
class _DummyRateLimit:
    """Minimal stand-in for RateLimit that never touches Redis."""

    _instance_dict: dict[str, "_DummyRateLimit"] = {}

    def __new__(cls, client_id: str, max_active_requests: int):
        # avoid singleton caching across tests
        instance = object.__new__(cls)
        return instance

    def __init__(self, client_id: str, max_active_requests: int) -> None:
        self.client_id = client_id
        self.max_active_requests = max_active_requests
        self._exited: list[str] = []

    @staticmethod
    def gen_request_key() -> str:
        return "dummy-request-id"

    def enter(self, request_id: str | None = None) -> str:
        return request_id or "dummy-request-id"

    def exit(self, request_id: str) -> None:
        self._exited.append(request_id)

    def generate(self, generator, request_id: str):
        return generator


_STREAM_REQUEST_ID = "dummy-request-id"


class _TrackingRateLimit(_DummyRateLimit):
    """Records occupancy with Redis-hash semantics: one slot per request id."""

    instances: list["_TrackingRateLimit"] = []

    def __init__(self, client_id: str, max_active_requests: int) -> None:
        super().__init__(client_id, max_active_requests)
        self.enter_calls: list[str] = []
        self.exit_calls: list[str] = []
        self.active: set[str] = set()
        self.instances.append(self)

    def enter(self, request_id: str | None = None) -> str:
        resolved_request_id = super().enter(request_id)
        self.enter_calls.append(resolved_request_id)
        self.active.add(resolved_request_id)
        return resolved_request_id

    def exit(self, request_id: str) -> None:
        super().exit(request_id)
        self.exit_calls.append(request_id)
        self.active.discard(request_id)

    def generate(self, generator: object, request_id: str) -> object:
        if isinstance(generator, Mapping):
            return generator
        return RateLimitGenerator(
            rate_limit=cast(RateLimit, self),
            generator=cast(Generator[str, None, None], generator),
            request_id=request_id,
        )


def _event_stream(*chunks: str, error: BaseException | None = None) -> Generator[str, None, None]:
    yield from chunks
    if error is not None:
        raise error


def _make_app(mode: AppMode | str, *, max_active_requests: int = 0) -> App:
    app = App(
        id="app-id",
        tenant_id="tenant-id",
        name="App",
        description="",
        mode=AppMode.CHAT if isinstance(mode, str) and mode == "invalid-mode" else mode,
        status=AppStatus.NORMAL,
        enable_site=False,
        enable_api=False,
        api_rpm=0,
        api_rph=0,
        max_active_requests=max_active_requests,
    )
    if mode == "invalid-mode":
        app.mode = mode  # type: ignore[assignment]
    return app


def _make_user() -> Account:
    user = Account(name="User", email="user@example.com")
    user.id = "user-id"
    return user


class _RealSessionTest:
    @pytest.fixture(autouse=True)
    def _bind_unbound_session(self, unbound_session: Session) -> None:
        self.session = unbound_session


def _make_workflow(
    *,
    workflow_id: str = "workflow-id",
    created_by: str = "owner-id",
    node_types: tuple[str, ...] = (),
) -> Workflow:
    return Workflow(
        id=workflow_id,
        tenant_id="tenant-id",
        app_id="app-id",
        type=WorkflowType.WORKFLOW,
        version=Workflow.VERSION_DRAFT,
        graph=json.dumps(
            {
                "nodes": [
                    {"id": f"node-{index}", "data": {"type": node_type}} for index, node_type in enumerate(node_types)
                ],
                "edges": [],
            }
        ),
        features={},
        created_by=created_by,
        environment_variables=[],
        conversation_variables=[],
    )


def _make_workflow_run(*, run_id: str, ended: bool) -> WorkflowRun:
    run = WorkflowRun(
        tenant_id="tenant-id",
        app_id="app-id",
        workflow_id="workflow-id",
        type=WorkflowType.WORKFLOW,
        triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
        version="published",
        status=WorkflowExecutionStatus.SUCCEEDED if ended else WorkflowExecutionStatus.RUNNING,
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="user-id",
    )
    run.id = run_id
    return run


# ---------------------------------------------------------------------------
# _build_streaming_task_on_subscribe
# ---------------------------------------------------------------------------
class _FakeTimer:
    def __init__(self, interval: float, function: Callable[[], bool]) -> None:
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


def _unexpected_timer(interval: float, function: Callable[[], bool]) -> _FakeTimer:
    raise AssertionError("streams must not create a fallback timer")


class TestBuildStreamingTaskOnSubscribe:
    def test_streams_starts_only_when_hook_is_invoked_without_creating_timer(
        self, monkeypatch: pytest.MonkeyPatch, config_overrides: Callable[..., None]
    ):
        config_overrides(PUBSUB_REDIS_CHANNEL_TYPE="streams")

        monkeypatch.setattr(ags_module.threading, "Timer", _unexpected_timer)
        called: list[int] = []

        on_subscribe = AppGenerateService._build_streaming_task_on_subscribe(lambda: called.append(1))

        assert called == []
        on_subscribe()
        on_subscribe()
        assert called == [1]

    @pytest.mark.parametrize("channel_type", ["pubsub", "sharded"])
    def test_pubsub_transports_keep_subscribe_hook_and_fallback_timer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        channel_type: str,
        config_overrides: Callable[..., None],
    ):
        config_overrides(PUBSUB_REDIS_CHANNEL_TYPE=channel_type)
        timers: list[_FakeTimer] = []

        def build_timer(interval: float, function: Callable[[], bool]) -> _FakeTimer:
            timer = _FakeTimer(interval, function)
            timers.append(timer)
            return timer

        monkeypatch.setattr(ags_module.threading, "Timer", build_timer)
        called: list[int] = []

        on_subscribe = AppGenerateService._build_streaming_task_on_subscribe(lambda: called.append(1))

        assert called == []
        assert len(timers) == 1
        assert timers[0].interval == ags_module.SSE_TASK_START_FALLBACK_MS / 1000.0
        assert timers[0].started is True

        on_subscribe()

        assert called == [1]
        assert timers[0].cancelled is True

    def test_pubsub_fallback_starts_task_if_hook_is_never_invoked(
        self, monkeypatch: pytest.MonkeyPatch, config_overrides: Callable[..., None]
    ):
        config_overrides(PUBSUB_REDIS_CHANNEL_TYPE="pubsub")
        timers: list[_FakeTimer] = []

        def build_timer(interval: float, function: Callable[[], bool]) -> _FakeTimer:
            timer = _FakeTimer(interval, function)
            timers.append(timer)
            return timer

        monkeypatch.setattr(ags_module.threading, "Timer", build_timer)
        called: list[int] = []
        on_subscribe = AppGenerateService._build_streaming_task_on_subscribe(lambda: called.append(1))

        assert timers[0].function() is True
        on_subscribe()
        assert called == [1]

    def test_streams_retries_after_enqueue_failure(
        self, monkeypatch: pytest.MonkeyPatch, config_overrides: Callable[..., None]
    ):
        config_overrides(PUBSUB_REDIS_CHANNEL_TYPE="streams")
        monkeypatch.setattr(ags_module.threading, "Timer", _unexpected_timer)
        call_count = 0

        def _bad():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")

        on_subscribe = AppGenerateService._build_streaming_task_on_subscribe(_bad)
        on_subscribe()
        assert call_count == 1
        on_subscribe()
        assert call_count == 2

    def test_concurrent_subscribe_only_starts_once(
        self, monkeypatch: pytest.MonkeyPatch, config_overrides: Callable[..., None]
    ):
        config_overrides(PUBSUB_REDIS_CHANNEL_TYPE="streams")
        monkeypatch.setattr(ags_module.threading, "Timer", _unexpected_timer)
        call_count = 0

        def _inc():
            nonlocal call_count
            call_count += 1

        cb = AppGenerateService._build_streaming_task_on_subscribe(_inc)
        threads = [threading.Thread(target=cb) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert call_count == 1


# ---------------------------------------------------------------------------
# _get_max_active_requests
# ---------------------------------------------------------------------------
class TestGetMaxActiveRequests:
    def test_both_zero_returns_zero(self, config_overrides: Callable[..., None]):
        config_overrides(APP_MAX_ACTIVE_REQUESTS=0, APP_DEFAULT_ACTIVE_REQUESTS=0)
        app = _make_app(AppMode.CHAT, max_active_requests=0)
        assert AppGenerateService._get_max_active_requests(app) == 0

    def test_app_limit_only(self, config_overrides: Callable[..., None]):
        config_overrides(APP_MAX_ACTIVE_REQUESTS=0, APP_DEFAULT_ACTIVE_REQUESTS=0)
        app = _make_app(AppMode.CHAT, max_active_requests=5)
        assert AppGenerateService._get_max_active_requests(app) == 5

    def test_config_limit_only(self, config_overrides: Callable[..., None]):
        config_overrides(APP_MAX_ACTIVE_REQUESTS=10, APP_DEFAULT_ACTIVE_REQUESTS=0)
        app = _make_app(AppMode.CHAT, max_active_requests=0)
        assert AppGenerateService._get_max_active_requests(app) == 10

    def test_both_non_zero_returns_min(self, config_overrides: Callable[..., None]):
        config_overrides(APP_MAX_ACTIVE_REQUESTS=20, APP_DEFAULT_ACTIVE_REQUESTS=0)
        app = _make_app(AppMode.CHAT, max_active_requests=5)
        assert AppGenerateService._get_max_active_requests(app) == 5

    def test_default_active_requests_used_when_app_has_none(self, config_overrides: Callable[..., None]):
        config_overrides(APP_MAX_ACTIVE_REQUESTS=0, APP_DEFAULT_ACTIVE_REQUESTS=15)
        app = _make_app(AppMode.CHAT, max_active_requests=0)
        assert AppGenerateService._get_max_active_requests(app) == 15


# ---------------------------------------------------------------------------
# generate – every AppMode branch
# ---------------------------------------------------------------------------
class TestGenerate(_RealSessionTest):
    """Tests for AppGenerateService.generate covering each mode."""

    @pytest.fixture(autouse=True)
    def _common(self, mocker: MockerFixture, config_overrides: Callable[..., None]):
        config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)
        mocker.patch("services.app_generate_service.RateLimit", _DummyRateLimit)

    # -- COMPLETION ---------------------------------------------------------
    def test_completion_mode(self, mocker: MockerFixture):
        gen_spy = mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.generate",
            return_value={"result": "ok"},
        )
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        result = AppGenerateService.generate(
            app_model=_make_app(AppMode.COMPLETION),
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=self.session,
        )
        assert result == {"result": "ok"}
        gen_spy.assert_called_once()

    # -- AGENT_CHAT via mode ------------------------------------------------
    def test_agent_chat_mode(self, mocker: MockerFixture):
        gen_spy = mocker.patch(
            "services.app_generate_service.AgentChatAppGenerator.generate",
            return_value={"result": "agent"},
        )
        mocker.patch(
            "services.app_generate_service.AgentChatAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        result = AppGenerateService.generate(
            app_model=_make_app(AppMode.AGENT_CHAT),
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=self.session,
        )
        assert result == {"result": "agent"}
        gen_spy.assert_called_once()

    # -- AGENT_CHAT via is_agent flag (non-AGENT_CHAT mode) -----------------
    def test_agent_via_is_agent_flag(self, mocker: MockerFixture):
        gen_spy = mocker.patch(
            "services.app_generate_service.AgentChatAppGenerator.generate",
            return_value={"result": "agent-via-flag"},
        )
        mocker.patch(
            "services.app_generate_service.AgentChatAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        app = _make_app(AppMode.CHAT)
        is_agent = mocker.patch.object(App, "is_agent_with_session", return_value=True)
        session = self.session
        result = AppGenerateService.generate(
            app_model=app,
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=session,
        )
        assert result == {"result": "agent-via-flag"}
        gen_spy.assert_called_once()
        is_agent.assert_called_once_with(session=session)

    # -- AGENT --------------------------------------------------------------
    def test_agent_mode_passes_session(self, mocker: MockerFixture):
        gen_spy = mocker.patch(
            "services.app_generate_service.AgentAppGenerator.generate",
            return_value={"result": "agent"},
        )
        mocker.patch(
            "services.app_generate_service.AgentAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        session = self.session

        result = AppGenerateService.generate(
            app_model=_make_app(AppMode.AGENT),
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=True,
            session=session,
        )

        assert result == {"result": "agent"}
        assert gen_spy.call_args.kwargs["session"] is session

    # -- CHAT ---------------------------------------------------------------
    def test_chat_mode(self, mocker: MockerFixture):
        gen_spy = mocker.patch(
            "services.app_generate_service.ChatAppGenerator.generate",
            return_value={"result": "chat"},
        )
        mocker.patch(
            "services.app_generate_service.ChatAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        app = _make_app(AppMode.CHAT)
        result = AppGenerateService.generate(
            app_model=app,
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=self.session,
        )
        assert result == {"result": "chat"}
        gen_spy.assert_called_once()

    # -- ADVANCED_CHAT blocking ---------------------------------------------
    def test_advanced_chat_blocking(self, mocker: MockerFixture):
        workflow = _make_workflow()
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)

        retrieve_spy = mocker.patch("services.app_generate_service.AdvancedChatAppGenerator.retrieve_events")
        gen_spy = mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator.generate",
            return_value={"result": "advanced-blocking"},
        )
        mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )

        session = self.session
        result = AppGenerateService.generate(
            app_model=_make_app(AppMode.ADVANCED_CHAT),
            user=_make_user(),
            args={"workflow_id": None, "query": "hi", "inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=session,
        )
        assert result == {"result": "advanced-blocking"}
        call_kwargs = gen_spy.call_args.kwargs
        assert call_kwargs.get("streaming") is False
        assert call_kwargs["session"] is session
        retrieve_spy.assert_not_called()

    # -- ADVANCED_CHAT streaming --------------------------------------------
    def test_advanced_chat_streaming(self, mocker: MockerFixture, config_overrides: Callable[..., None]):
        config_overrides(PUBSUB_REDIS_CHANNEL_TYPE="streams")
        workflow = _make_workflow()
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        mocker.patch(
            "services.app_generate_service.AppExecutionParams.new",
            return_value=MagicMock(workflow_run_id="wfr-1", model_dump_json=MagicMock(return_value="{}")),
        )
        delay_spy = mocker.patch("services.app_generate_service.workflow_based_app_execution_task.delay")
        gen_instance = MagicMock()
        gen_instance.retrieve_events.return_value = iter([])
        gen_instance.convert_to_event_stream.side_effect = lambda x: x
        mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator",
            return_value=gen_instance,
        )

        result = AppGenerateService.generate(
            app_model=_make_app(AppMode.ADVANCED_CHAT),
            user=_make_user(),
            args={"workflow_id": None, "query": "hi", "inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=True,
            session=self.session,
        )
        # In streaming mode it should go through retrieve_events, not generate
        gen_instance.retrieve_events.assert_called_once()
        # Dispatch is gated on subscribe; simulate the SSE layer entering the
        # subscription, which is what actually invokes on_subscribe.
        on_subscribe = gen_instance.retrieve_events.call_args.kwargs["on_subscribe"]
        on_subscribe()
        delay_spy.assert_called_once()

    # -- WORKFLOW blocking --------------------------------------------------
    def test_workflow_blocking(self, mocker: MockerFixture):
        workflow = _make_workflow()
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        gen_spy = mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.generate",
            return_value={"result": "workflow-blocking"},
        )
        mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )

        session = self.session
        result = AppGenerateService.generate(
            app_model=_make_app(AppMode.WORKFLOW),
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=session,
        )
        assert result == {"result": "workflow-blocking"}
        call_kwargs = gen_spy.call_args.kwargs
        assert call_kwargs.get("pause_state_config") is not None
        assert call_kwargs["pause_state_config"].state_owner_user_id == "owner-id"

    @pytest.mark.parametrize(
        "invoke_from",
        [InvokeFrom.OPENAPI, InvokeFrom.SERVICE_API, InvokeFrom.WEB_APP],
    )
    @pytest.mark.parametrize("node_type", ["trigger-plugin", "trigger-schedule", "trigger-webhook"])
    def test_trigger_workflow_rejects_manual_service_surfaces(
        self,
        invoke_from: InvokeFrom,
        node_type: str,
        mocker: MockerFixture,
    ) -> None:
        workflow = _make_workflow(node_types=(node_type,))
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        generate = mocker.patch("services.app_generate_service.WorkflowAppGenerator.generate")

        with pytest.raises(TriggerWorkflowServiceModeUnavailableError):
            AppGenerateService.generate(
                app_model=_make_app(AppMode.WORKFLOW),
                user=_make_user(),
                args={"inputs": {}},
                invoke_from=invoke_from,
                streaming=False,
                session=MagicMock(),
            )

        generate.assert_not_called()

    def test_trigger_workflow_allows_trigger_execution(self, mocker: MockerFixture) -> None:
        workflow = _make_workflow(node_types=("trigger-webhook",))
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        generate = mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.generate",
            return_value={"result": "trigger"},
        )
        mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.convert_to_event_stream",
            side_effect=lambda value: value,
        )

        result = AppGenerateService.generate(
            app_model=_make_app(AppMode.WORKFLOW),
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.TRIGGER,
            streaming=False,
            session=MagicMock(),
        )

        assert result == {"result": "trigger"}
        generate.assert_called_once()

    def test_specific_start_workflow_version_remains_runnable(self, mocker: MockerFixture) -> None:
        workflow_id = str(uuid.uuid4())
        workflow = _make_workflow(workflow_id=workflow_id, node_types=("start",))
        get_workflow = mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.generate",
            return_value={"result": "version"},
        )
        mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.convert_to_event_stream",
            side_effect=lambda value: value,
        )
        app = _make_app(AppMode.WORKFLOW)
        session = MagicMock()

        result = AppGenerateService.generate(
            app_model=app,
            user=_make_user(),
            args={"inputs": {}, "workflow_id": workflow_id},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=session,
        )

        assert result == {"result": "version"}
        get_workflow.assert_called_once_with(app, InvokeFrom.SERVICE_API, workflow_id, session=session)

    # -- WORKFLOW streaming -------------------------------------------------
    def test_workflow_streaming(self, mocker: MockerFixture, config_overrides: Callable[..., None]):
        config_overrides(PUBSUB_REDIS_CHANNEL_TYPE="streams")
        workflow = _make_workflow()
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        mocker.patch(
            "services.app_generate_service.AppExecutionParams.new",
            return_value=MagicMock(workflow_run_id="wfr-2", model_dump_json=MagicMock(return_value="{}")),
        )
        delay_spy = mocker.patch("services.app_generate_service.workflow_based_app_execution_task.delay")
        retrieve_spy = mocker.patch(
            "services.app_generate_service.MessageBasedAppGenerator.retrieve_events",
            return_value=iter([]),
        )
        mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )

        result = AppGenerateService.generate(
            app_model=_make_app(AppMode.WORKFLOW),
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=True,
            session=self.session,
        )
        retrieve_spy.assert_called_once()
        # Dispatch is gated on subscribe; simulate the SSE layer entering the
        # subscription, which is what actually invokes on_subscribe.
        on_subscribe = retrieve_spy.call_args.kwargs["on_subscribe"]
        on_subscribe()
        delay_spy.assert_called_once()

    # -- Invalid mode -------------------------------------------------------
    def test_invalid_mode_raises(self, mocker: MockerFixture):
        app = _make_app("invalid-mode")
        with pytest.raises(ValueError, match="Invalid app mode"):
            AppGenerateService.generate(
                app_model=app,
                user=_make_user(),
                args={},
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=False,
                session=self.session,
            )


# ---------------------------------------------------------------------------
# generate – streaming rate limit
# ---------------------------------------------------------------------------
class TestGenerateStreamingRateLimit(_RealSessionTest):
    """The app active-request slot stays occupied until a streaming run actually closes."""

    @pytest.fixture(autouse=True)
    def _common(self, mocker: MockerFixture, config_overrides: Callable[..., None]) -> None:
        config_overrides(
            DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY,
            PUBSUB_REDIS_CHANNEL_TYPE="streams",
        )
        _TrackingRateLimit.instances.clear()
        mocker.patch("services.app_generate_service.RateLimit", _TrackingRateLimit)

    def _open(
        self,
        mocker: MockerFixture,
        mode: AppMode,
        events: Generator[str, None, None],
    ) -> tuple[RateLimitGenerator, _TrackingRateLimit]:
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=_make_workflow())
        mocker.patch(
            "services.app_generate_service.AppExecutionParams.new",
            return_value=MagicMock(workflow_run_id="wfr-stream", model_dump_json=MagicMock(return_value="{}")),
        )
        mocker.patch("services.app_generate_service.workflow_based_app_execution_task.delay")
        if mode is AppMode.ADVANCED_CHAT:
            generator = MagicMock()
            generator.retrieve_events.return_value = events
            generator.convert_to_event_stream.side_effect = lambda value: value
            mocker.patch("services.app_generate_service.AdvancedChatAppGenerator", return_value=generator)
            args: Mapping[str, object] = {"workflow_id": None, "query": "hi", "inputs": {}}
        elif mode is AppMode.WORKFLOW:
            mocker.patch(
                "services.app_generate_service.MessageBasedAppGenerator.retrieve_events",
                return_value=events,
            )
            mocker.patch(
                "services.app_generate_service.WorkflowAppGenerator.convert_to_event_stream",
                side_effect=lambda value: value,
            )
            args = {"inputs": {}}
        else:
            raise AssertionError(f"unsupported streaming mode {mode}")

        stream = AppGenerateService.generate(
            app_model=_make_app(mode, max_active_requests=2),
            user=_make_user(),
            args=args,
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=True,
            session=self.session,
        )
        assert isinstance(stream, RateLimitGenerator)
        return stream, _TrackingRateLimit.instances[-1]

    def _assert_still_active(self, rate_limit: _TrackingRateLimit) -> None:
        assert rate_limit.enter_calls == [_STREAM_REQUEST_ID]
        assert rate_limit.exit_calls == []
        assert rate_limit.active == {_STREAM_REQUEST_ID}

    def _assert_released_once(self, rate_limit: _TrackingRateLimit) -> None:
        assert rate_limit.exit_calls == [_STREAM_REQUEST_ID]
        assert rate_limit.active == set()

    @pytest.mark.parametrize("mode", [AppMode.WORKFLOW, AppMode.ADVANCED_CHAT])
    def test_normal_close_releases_slot_once(self, mocker: MockerFixture, mode: AppMode) -> None:
        stream, rate_limit = self._open(mocker, mode, _event_stream("one", "two"))
        self._assert_still_active(rate_limit)

        assert next(stream) == "one"
        self._assert_still_active(rate_limit)
        assert list(stream) == ["two"]
        self._assert_released_once(rate_limit)

        stream.close()
        del stream
        gc.collect()
        self._assert_released_once(rate_limit)

    @pytest.mark.parametrize("mode", [AppMode.WORKFLOW, AppMode.ADVANCED_CHAT])
    def test_error_mid_stream_releases_slot_once(self, mocker: MockerFixture, mode: AppMode) -> None:
        stream, rate_limit = self._open(
            mocker,
            mode,
            _event_stream("one", error=RuntimeError("boom")),
        )
        self._assert_still_active(rate_limit)

        assert next(stream) == "one"
        self._assert_still_active(rate_limit)
        with pytest.raises(RuntimeError, match="boom"):
            next(stream)
        self._assert_released_once(rate_limit)

        stream.close()
        del stream
        gc.collect()
        self._assert_released_once(rate_limit)

    @pytest.mark.parametrize("mode", [AppMode.WORKFLOW, AppMode.ADVANCED_CHAT])
    def test_client_disconnect_releases_slot_once(self, mocker: MockerFixture, mode: AppMode) -> None:
        stream, rate_limit = self._open(mocker, mode, _event_stream("one", "two"))
        self._assert_still_active(rate_limit)

        assert next(stream) == "one"
        self._assert_still_active(rate_limit)
        stream.close()
        self._assert_released_once(rate_limit)

        stream.close()
        with pytest.raises(StopIteration):
            next(stream)
        del stream
        gc.collect()
        self._assert_released_once(rate_limit)

    @pytest.mark.parametrize("mode", [AppMode.WORKFLOW, AppMode.ADVANCED_CHAT])
    def test_discarded_generator_releases_slot_once(self, mocker: MockerFixture, mode: AppMode) -> None:
        stream, rate_limit = self._open(mocker, mode, _event_stream("one", "two"))
        self._assert_still_active(rate_limit)

        assert next(stream) == "one"
        self._assert_still_active(rate_limit)
        del stream
        gc.collect()
        self._assert_released_once(rate_limit)


# ---------------------------------------------------------------------------
# generate – billing / quota
# ---------------------------------------------------------------------------
class TestGenerateBilling(_RealSessionTest):
    @pytest.fixture(autouse=True)
    def _common(self, mocker: MockerFixture):
        mocker.patch("services.app_generate_service.RateLimit", _DummyRateLimit)

    def test_cloud_edition_consumes_quota(self, mocker: MockerFixture, config_overrides: Callable[..., None]):
        config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
        quota_charge = MagicMock()
        reserve_mock = mocker.patch(
            "services.app_generate_service.QuotaService.reserve",
            return_value=quota_charge,
        )
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.generate",
            return_value={"ok": True},
        )
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )

        AppGenerateService.generate(
            app_model=_make_app(AppMode.COMPLETION),
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=self.session,
        )
        reserve_mock.assert_called_once_with(QuotaType.WORKFLOW, "tenant-id")
        quota_charge.commit.assert_called_once()

    def test_billing_quota_exceeded_raises_rate_limit_error(
        self, mocker: MockerFixture, config_overrides: Callable[..., None]
    ):
        from services.errors.app import QuotaExceededError
        from services.errors.llm import InvokeRateLimitError

        config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
        mocker.patch(
            "services.app_generate_service.QuotaService.reserve",
            side_effect=QuotaExceededError(feature="workflow", tenant_id="t", required=1),
        )

        with pytest.raises(InvokeRateLimitError):
            AppGenerateService.generate(
                app_model=_make_app(AppMode.COMPLETION),
                user=_make_user(),
                args={"inputs": {}},
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=False,
                session=self.session,
            )

    def test_exception_refunds_quota_and_exits_rate_limit(
        self, mocker: MockerFixture, config_overrides: Callable[..., None]
    ):
        config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
        quota_charge = MagicMock()
        mocker.patch(
            "services.app_generate_service.QuotaService.reserve",
            return_value=quota_charge,
        )
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.generate",
            side_effect=RuntimeError("boom"),
        )
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )

        with pytest.raises(RuntimeError, match="boom"):
            AppGenerateService.generate(
                app_model=_make_app(AppMode.COMPLETION),
                user=_make_user(),
                args={"inputs": {}},
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=False,
                session=self.session,
            )
        quota_charge.refund.assert_called_once()

    def test_rate_limit_exit_called_in_finally_for_blocking(
        self, mocker: MockerFixture, config_overrides: Callable[..., None]
    ):
        """For non-streaming (blocking) calls, rate_limit.exit should be called in finally."""
        config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.COMMUNITY)

        exit_calls: list[str] = []

        class _TrackingRateLimit(_DummyRateLimit):
            def exit(self, request_id: str) -> None:
                exit_calls.append(request_id)

        mocker.patch("services.app_generate_service.RateLimit", _TrackingRateLimit)
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.generate",
            return_value={"ok": True},
        )
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )

        AppGenerateService.generate(
            app_model=_make_app(AppMode.COMPLETION),
            user=_make_user(),
            args={"inputs": {}},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=False,
            session=self.session,
        )
        # exit is called in finally block for non-streaming
        assert exit_calls == ["dummy-request-id"]

    def test_blocking_failure_exits_rate_limit_once(self, mocker: MockerFixture, config_overrides: Callable[..., None]):
        config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
        quota_charge = MagicMock()
        mocker.patch(
            "services.app_generate_service.QuotaService.reserve",
            return_value=quota_charge,
        )
        exit_calls: list[str] = []

        class _TrackingRateLimit(_DummyRateLimit):
            def exit(self, request_id: str) -> None:
                exit_calls.append(request_id)

        mocker.patch("services.app_generate_service.RateLimit", _TrackingRateLimit)
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.generate",
            side_effect=RuntimeError("boom"),
        )

        with pytest.raises(RuntimeError, match="boom"):
            AppGenerateService.generate(
                app_model=_make_app(AppMode.COMPLETION),
                user=_make_user(),
                args={"inputs": {}},
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=False,
                session=self.session,
            )

        quota_charge.refund.assert_called_once()
        assert exit_calls == ["dummy-request-id"]

    def test_streaming_failure_exits_rate_limit_once(
        self, mocker: MockerFixture, config_overrides: Callable[..., None]
    ):
        config_overrides(DEPLOYMENT_EDITION=DeploymentEdition.CLOUD)
        quota_charge = MagicMock()
        mocker.patch(
            "services.app_generate_service.QuotaService.reserve",
            return_value=quota_charge,
        )
        exit_calls: list[str] = []

        class _TrackingRateLimit(_DummyRateLimit):
            def exit(self, request_id: str) -> None:
                exit_calls.append(request_id)

        mocker.patch("services.app_generate_service.RateLimit", _TrackingRateLimit)
        mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.generate",
            side_effect=RuntimeError("boom"),
        )

        with pytest.raises(RuntimeError, match="boom"):
            AppGenerateService.generate(
                app_model=_make_app(AppMode.COMPLETION),
                user=_make_user(),
                args={"inputs": {}},
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=True,
                session=self.session,
            )

        quota_charge.refund.assert_called_once()
        assert exit_calls == ["dummy-request-id"]


# ---------------------------------------------------------------------------
# _get_workflow
# ---------------------------------------------------------------------------
class TestGetWorkflow(_RealSessionTest):
    def test_debugger_fetches_draft(self, mocker: MockerFixture):
        draft_wf = _make_workflow()
        ws = MagicMock()
        ws.get_draft_workflow.return_value = draft_wf
        mocker.patch("services.app_generate_service.WorkflowService", return_value=ws)

        result = AppGenerateService._get_workflow(
            _make_app(AppMode.WORKFLOW), InvokeFrom.DEBUGGER, session=self.session
        )
        assert result is draft_wf
        ws.get_draft_workflow.assert_called_once()

    def test_debugger_raises_when_no_draft(self, mocker: MockerFixture):
        ws = MagicMock()
        ws.get_draft_workflow.return_value = None
        mocker.patch("services.app_generate_service.WorkflowService", return_value=ws)

        with pytest.raises(ValueError, match="Workflow not initialized"):
            AppGenerateService._get_workflow(_make_app(AppMode.WORKFLOW), InvokeFrom.DEBUGGER, session=self.session)

    def test_non_debugger_fetches_published(self, mocker: MockerFixture):
        pub_wf = _make_workflow()
        ws = MagicMock()
        ws.get_published_workflow.return_value = pub_wf
        mocker.patch("services.app_generate_service.WorkflowService", return_value=ws)

        result = AppGenerateService._get_workflow(
            _make_app(AppMode.WORKFLOW), InvokeFrom.SERVICE_API, session=self.session
        )
        assert result is pub_wf
        ws.get_published_workflow.assert_called_once()

    def test_non_debugger_raises_when_no_published(self, mocker: MockerFixture):
        ws = MagicMock()
        ws.get_published_workflow.return_value = None
        mocker.patch("services.app_generate_service.WorkflowService", return_value=ws)

        with pytest.raises(ValueError, match="Workflow not published"):
            AppGenerateService._get_workflow(_make_app(AppMode.WORKFLOW), InvokeFrom.SERVICE_API, session=self.session)

    def test_specific_workflow_id_valid_uuid(self, mocker: MockerFixture):
        valid_uuid = str(uuid.uuid4())
        specific_wf = _make_workflow(workflow_id=valid_uuid)
        ws = MagicMock()
        ws.get_published_workflow_by_id.return_value = specific_wf
        mocker.patch("services.app_generate_service.WorkflowService", return_value=ws)

        result = AppGenerateService._get_workflow(
            _make_app(AppMode.WORKFLOW),
            InvokeFrom.SERVICE_API,
            workflow_id=valid_uuid,
            session=self.session,
        )
        assert result is specific_wf
        ws.get_published_workflow_by_id.assert_called_once()

    def test_specific_workflow_id_invalid_uuid(self, mocker: MockerFixture):
        ws = MagicMock()
        mocker.patch("services.app_generate_service.WorkflowService", return_value=ws)

        with pytest.raises(WorkflowIdFormatError):
            AppGenerateService._get_workflow(
                _make_app(AppMode.WORKFLOW),
                InvokeFrom.SERVICE_API,
                workflow_id="not-a-uuid",
                session=self.session,
            )

    def test_specific_workflow_id_not_found(self, mocker: MockerFixture):
        valid_uuid = str(uuid.uuid4())
        ws = MagicMock()
        ws.get_published_workflow_by_id.return_value = None
        mocker.patch("services.app_generate_service.WorkflowService", return_value=ws)

        with pytest.raises(WorkflowNotFoundError):
            AppGenerateService._get_workflow(
                _make_app(AppMode.WORKFLOW),
                InvokeFrom.SERVICE_API,
                workflow_id=valid_uuid,
                session=self.session,
            )


# ---------------------------------------------------------------------------
# generate_single_iteration
# ---------------------------------------------------------------------------
class TestGenerateSingleIteration(_RealSessionTest):
    def test_advanced_chat_mode(self, mocker: MockerFixture):
        workflow = _make_workflow()
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        gen_spy = mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        iter_spy = mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator.single_iteration_generate",
            return_value={"event": "iteration"},
        )
        app = _make_app(AppMode.ADVANCED_CHAT)
        session = self.session
        result = AppGenerateService.generate_single_iteration(
            app_model=app,
            user=_make_user(),
            node_id="n1",
            args={"k": "v"},
            session=session,
        )
        iter_spy.assert_called_once()
        assert iter_spy.call_args.kwargs["session"] is session
        assert result == {"event": "iteration"}

    def test_workflow_mode(self, mocker: MockerFixture):
        workflow = _make_workflow()
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        iter_spy = mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.single_iteration_generate",
            return_value={"event": "wf-iteration"},
        )
        app = _make_app(AppMode.WORKFLOW)
        session = self.session
        result = AppGenerateService.generate_single_iteration(
            app_model=app,
            user=_make_user(),
            node_id="n1",
            args={"k": "v"},
            session=session,
        )
        iter_spy.assert_called_once()
        assert iter_spy.call_args.kwargs["session"] is session
        assert result == {"event": "wf-iteration"}

    def test_invalid_mode_raises(self, mocker: MockerFixture):
        app = _make_app(AppMode.CHAT)
        with pytest.raises(ValueError, match="Invalid app mode"):
            AppGenerateService.generate_single_iteration(
                app_model=app, user=_make_user(), node_id="n1", args={}, session=self.session
            )


# ---------------------------------------------------------------------------
# generate_single_loop
# ---------------------------------------------------------------------------
class TestGenerateSingleLoop(_RealSessionTest):
    def test_advanced_chat_mode(self, mocker: MockerFixture):
        workflow = _make_workflow()
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        loop_spy = mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator.single_loop_generate",
            return_value={"event": "loop"},
        )
        app = _make_app(AppMode.ADVANCED_CHAT)
        session = self.session
        result = AppGenerateService.generate_single_loop(
            app_model=app,
            user=_make_user(),
            node_id="n1",
            args=MagicMock(),
            session=session,
        )
        loop_spy.assert_called_once()
        assert loop_spy.call_args.kwargs["session"] is session
        assert result == {"event": "loop"}

    def test_workflow_mode(self, mocker: MockerFixture):
        workflow = _make_workflow()
        mocker.patch.object(AppGenerateService, "_get_workflow", return_value=workflow)
        mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator.convert_to_event_stream",
            side_effect=lambda x: x,
        )
        loop_spy = mocker.patch(
            "services.app_generate_service.WorkflowAppGenerator.single_loop_generate",
            return_value={"event": "wf-loop"},
        )
        app = _make_app(AppMode.WORKFLOW)
        session = self.session
        result = AppGenerateService.generate_single_loop(
            app_model=app,
            user=_make_user(),
            node_id="n1",
            args=MagicMock(),
            session=session,
        )
        loop_spy.assert_called_once()
        assert loop_spy.call_args.kwargs["session"] is session
        assert result == {"event": "wf-loop"}

    def test_invalid_mode_raises(self, mocker: MockerFixture):
        app = _make_app(AppMode.COMPLETION)
        with pytest.raises(ValueError, match="Invalid app mode"):
            AppGenerateService.generate_single_loop(
                app_model=app, user=_make_user(), node_id="n1", args=MagicMock(), session=self.session
            )


# ---------------------------------------------------------------------------
# generate_more_like_this
# ---------------------------------------------------------------------------
class TestGenerateMoreLikeThis(_RealSessionTest):
    def test_delegates_to_completion_generator(self, mocker: MockerFixture):
        gen_spy = mocker.patch(
            "services.app_generate_service.CompletionAppGenerator.generate_more_like_this",
            return_value={"result": "similar"},
        )
        session = self.session
        result = AppGenerateService.generate_more_like_this(
            app_model=_make_app(AppMode.COMPLETION),
            user=_make_user(),
            message_id="msg-1",
            invoke_from=InvokeFrom.SERVICE_API,
            session=session,
            streaming=True,
        )
        assert result == {"result": "similar"}
        gen_spy.assert_called_once()
        assert gen_spy.call_args.kwargs["session"] is session
        assert gen_spy.call_args.kwargs["stream"] is True


# ---------------------------------------------------------------------------
# get_response_generator
# ---------------------------------------------------------------------------
class TestGetResponseGenerator:
    def test_non_ended_workflow_run(self, mocker: MockerFixture):
        app = _make_app(AppMode.ADVANCED_CHAT)
        workflow_run = _make_workflow_run(run_id="run-1", ended=False)

        gen_instance = MagicMock()
        gen_instance.retrieve_events.return_value = iter([{"event": "started"}])
        gen_instance.convert_to_event_stream.side_effect = lambda x: x
        mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator",
            return_value=gen_instance,
        )

        result = AppGenerateService.get_response_generator(app_model=app, workflow_run=workflow_run)
        gen_instance.retrieve_events.assert_called_once()

    def test_ended_workflow_run_still_returns_generator(self, mocker: MockerFixture):
        """Even when the run is ended, the current code still returns a generator (TODO branch)."""
        app = _make_app(AppMode.WORKFLOW)
        workflow_run = _make_workflow_run(run_id="run-2", ended=True)

        gen_instance = MagicMock()
        gen_instance.retrieve_events.return_value = iter([])
        gen_instance.convert_to_event_stream.side_effect = lambda x: x
        mocker.patch(
            "services.app_generate_service.AdvancedChatAppGenerator",
            return_value=gen_instance,
        )

        result = AppGenerateService.get_response_generator(app_model=app, workflow_run=workflow_run)
        # current impl falls through the TODO and still creates a generator
        gen_instance.retrieve_events.assert_called_once()
