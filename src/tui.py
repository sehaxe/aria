"""Aria Control Room — Textual TUI for training & recurrent thinking.

Gradio-free dashboard driving train.controller.TrainController and (lazily)
the thinking path from think.py. Live loss / tok-s sparklines, a breathing
status badge, and a thinking-depth meter.

Tabs:
  Training  — Start / Pause / Stop / Load 29m; status badge + progress,
              colored metrics, live loss & throughput sparklines, log feed.
  Thinking  — Prompt -> generate -> streamed output colored by thinking depth,
              with a live depth meter.
  Config    — Read-only view of configs/29m.yaml.

Bindings: q=quit, s=start, p=pause/resume toggle, x=stop.
"""
import sys
from pathlib import Path

# ponytail: src/ layout — put src/ on path so `from train.controller import ...`
# works whether launched as `python src/tui.py` or `python -m tui`.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
)

from train.controller import TrainController


CONFIG_PATH = str(_SRC.parent / "configs" / "29m.yaml")

ARIA_CSS = """
Screen {
    background: #060a0f;
    color: #cdd9e5;
}

Header {
    background: #0b121b;
    color: #2dd4bf;
    text-style: bold;
}

Footer {
    background: #0b121b;
    color: #6b7c8c;
}

TabbedContent {
    background: #060a0f;
}

TabPane {
    padding: 1 2;
}

.hero {
    background: #0b121b;
    border: round #16344a;
    color: #2dd4bf;
    text-align: center;
    padding: 1 2;
    margin: 0 0 1 0;
}

.card {
    background: #0b121b;
    border: round #1b2a3a;
    padding: 1 2;
    margin: 0 1 1 0;
}

.card-title {
    color: #2dd4bf;
    text-style: bold;
    margin: 0 0 1 0;
}

#status_badge {
    color: #6b7c8c;
    text-style: bold;
    padding: 0 2;
    margin: 0 0 1 0;
}

#status_badge.-running { color: #2dd4bf; text-style: bold; }
#status_badge.-paused  { color: #ff8a3d; text-style: bold; }
#status_badge.-stopped { color: #ff5c5c; }
#status_badge.-error   { color: #ff5c5c; text-style: bold reverse; }
#status_badge.-done    { color: #34e89e; }
#status_badge.-idle    { color: #6b7c8c; }

#progress {
    margin: 1 0;
}

ProgressBar > Bar {
    color: #2dd4bf;
}

Button {
    margin: 0 1 0 0;
    min-width: 12;
}

Button.-primary {
    background: #0b3d39;
    color: #2dd4bf;
    border: tall #0f5f57;
}

Button.-heat {
    background: #3a200c;
    color: #ff8a3d;
    border: tall #5a3413;
}

Button.-danger {
    background: #3a1313;
    color: #ff5c5c;
    border: tall #5a1f1f;
}

#metrics {
    height: auto;
    margin: 1 0;
}

#loss_spark {
    color: #ff8a3d;
    height: 5;
}

#tps_spark {
    color: #2dd4bf;
    height: 5;
}

.spark-label {
    color: #6b7c8c;
    width: 9;
}

#logs, #think_out, #cfg_view {
    background: #070d13;
    border: round #1b2a3a;
    height: 100%;
    min-height: 12;
}

#depth_meter {
    margin: 1 0;
}

ProgressBar#depth_meter > Bar {
    color: #ff8a3d;
}

Input {
    background: #070d13;
    border: round #1b2a3a;
    color: #cdd9e5;
}

Label {
    color: #6b7c8c;
}
"""


STATUS_LABELS = {
    "idle":    "● IDLE",
    "running": "▶ RUNNING",
    "paused":  "⏸ PAUSED",
    "done":    "✓ DONE",
    "stopped": "■ STOPPED",
    "error":   "✕ ERROR",
}
STATUS_CLASSES = {k: f"-{k}" for k in STATUS_LABELS}


def _read_cfg():
    try:
        with open(CONFIG_PATH) as f:
            return f.read()
    except Exception as e:
        return f"# config not found: {e}"


class AriaTUI(App):
    """Aria Control Room — Textual dashboard."""

    CSS = ARIA_CSS
    TITLE = "ARIA"
    SUB_TITLE = "recurrent ternary LLM · control room"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "start", "Start"),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("x", "stop", "Stop"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctrl = TrainController(ckpt_dir=str(_SRC.parent / "checkpoints"))
        self._last_log_len = 0
        self._last_metric_step = -1
        self._loss_hist: list[float] = []
        self._tps_hist: list[float] = []
        self._pulse_state = 1.0
        self._logs_widget: RichLog | None = None
        self._metrics_widget: Static | None = None
        self._status_widget: Static | None = None
        self._progress_widget: ProgressBar | None = None
        self._loss_spark: Sparkline | None = None
        self._tps_spark: Sparkline | None = None

    # -- compose ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="training"):
            with TabPane("Training", id="training"):
                yield from self._compose_training()
            with TabPane("Thinking", id="thinking"):
                yield from self._compose_thinking()
            with TabPane("Config", id="config"):
                yield from self._compose_config()
        yield Footer()

    def _compose_training(self) -> ComposeResult:
        yield Static(
            "[bold #2dd4bf]ARIA[/bold #2dd4bf] · [bold]CONTROL ROOM[/bold]\n"
            "[dim]recurrent ternary LLM — training & thinking console[/dim]",
            id="hero", classes="hero", markup=True,
        )
        with Horizontal():
            with Vertical(classes="card"):
                yield Static("CONTROL", classes="card-title")
                yield Static("● IDLE", id="status_badge", classes="-idle")
                yield ProgressBar(total=100, id="progress", show_eta=False)
                with Horizontal():
                    yield Button("▶ Start",    id="btn_start",  classes="-primary")
                    yield Button("⏸ Pause",    id="btn_pause",  classes="-heat")
                    yield Button("■ Stop",     id="btn_stop",   classes="-danger")
                    yield Button("⚙ Load 29m", id="btn_load29")
            with Vertical(classes="card"):
                yield Static("LIVE METRICS", classes="card-title")
                yield Static("", id="metrics")
                with Horizontal():
                    yield Label("loss", classes="spark-label")
                    yield Sparkline(data=[0.0, 0.0], id="loss_spark")
                with Horizontal():
                    yield Label("tok/s", classes="spark-label")
                    yield Sparkline(data=[0.0, 0.0], id="tps_spark")
        with Vertical(classes="card"):
            yield Static("LIVE LOGS", classes="card-title")
            yield RichLog(id="logs", highlight=True, markup=False, wrap=False, auto_scroll=True)

    def _compose_thinking(self) -> ComposeResult:
        with Vertical(classes="card"):
            yield Static("PROMPT", classes="card-title")
            yield Input(placeholder="e.g. Count the bolts in the box...",
                        id="prompt_input")
            with Horizontal():
                yield Input(value="128", id="think_maxlen")
                yield Input(value="0.7", id="think_temp")
                yield Button("Run thinking", id="btn_think", classes="-primary")
            yield Label("[left] max bytes  [right] temperature")
        with Vertical(classes="card"):
            yield Static("MODEL OUTPUT", classes="card-title")
            yield ProgressBar(total=100, id="depth_meter", show_eta=False)
            yield RichLog(id="think_out", markup=True, wrap=True, auto_scroll=True)

    def _compose_config(self) -> ComposeResult:
        with Vertical(classes="card"):
            yield Static(f"{CONFIG_PATH}", classes="card-title")
            with VerticalScroll():
                yield Static(_read_cfg(), id="cfg_view")

    # -- lifecycle -------------------------------------------------------

    def on_mount(self) -> None:
        self._logs_widget = self.query_one("#logs", RichLog)
        self._metrics_widget = self.query_one("#metrics", Static)
        self._status_widget = self.query_one("#status_badge", Static)
        self._progress_widget = self.query_one("#progress", ProgressBar)
        self._loss_spark = self.query_one("#loss_spark", Sparkline)
        self._tps_spark = self.query_one("#tps_spark", Sparkline)
        self.set_interval(0.5, self._poll)
        self.set_interval(1.1, self._pulse)
        self._logs_widget.write(
            "[dim]Aria Control Room online. s=start · p=pause · x=stop · q=quit[/dim]")

    # -- polling ---------------------------------------------------------

    def _poll(self) -> None:
        raw = self.ctrl.get_logs()
        lines = raw.split("\n") if raw else []
        if len(lines) > self._last_log_len:
            for line in lines[self._last_log_len:]:
                if line:
                    self._logs_widget.write(line)
            self._last_log_len = len(lines)

        m = self.ctrl.get_metrics()
        if m:
            step, loss, tok_s, mem = m[-1]
            if step != self._last_metric_step:
                self._loss_hist.append(loss)
                self._tps_hist.append(tok_s)
                if len(self._loss_hist) > 100:
                    self._loss_hist.pop(0)
                    self._tps_hist.pop(0)
                self._loss_spark.data = self._loss_hist if len(self._loss_hist) > 1 else [loss, loss]
                self._tps_spark.data = self._tps_hist if len(self._tps_hist) > 1 else [tok_s, tok_s]
                self._metrics_widget.update(self._fmt_metrics(m))
                self._last_metric_step = step

        st = self.ctrl.get_state()
        label = STATUS_LABELS.get(st["status"], st["status"])
        self._status_widget.update(f"{label}   step {st['step']}/{st['total']}")
        for cls in STATUS_CLASSES.values():
            self._status_widget.remove_class(cls)
        self._status_widget.add_class(STATUS_CLASSES.get(st["status"], "-idle"))
        pct = max(0.0, min(1.0, float(st["frac"]))) * 100.0
        self._progress_widget.update(total=100, progress=pct)

    def _pulse(self) -> None:
        if self.ctrl.get_state()["status"] == "running":
            self._pulse_state = 0.45 if self._pulse_state > 0.7 else 1.0
            self._status_widget.animate("opacity", self._pulse_state, duration=0.9)
        elif self._status_widget.styles.opacity != 1.0:
            self._status_widget.styles.opacity = 1.0

    @staticmethod
    def _fmt_metrics(m) -> str:
        step, loss, tok_s, mem = m[-1]
        loss_col = "#34e89e" if loss < 1.0 else ("#ff8a3d" if loss < 2.0 else "#ff5c5c")
        return (
            f"[#6b7c8c]step[/] [#2dd4bf]{step}[/]    "
            f"[#6b7c8c]loss[/] [{loss_col}]{loss:.4f}[/]    "
            f"[#6b7c8c]tok/s[/] [#2dd4bf]{tok_s:.0f}[/]    "
            f"[#6b7c8c]mem[/] [#ff8a3d]{mem:.2f}GB[/]"
        )

    # -- actions ---------------------------------------------------------

    def action_start(self) -> None:
        self._start_training()

    def action_toggle_pause(self) -> None:
        st = self.ctrl.get_state()
        if st["status"] == "paused":
            self.ctrl.resume()
        else:
            self.ctrl.pause()

    def action_stop(self) -> None:
        self.ctrl.stop()

    # -- button handlers -------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn_start":
            self._start_training()
        elif bid == "btn_pause":
            self.action_toggle_pause()
        elif bid == "btn_stop":
            self.ctrl.stop()
        elif bid == "btn_load29":
            self._load_29m()
        elif bid == "btn_think":
            self._run_thinking()

    def _load_29m(self) -> None:
        # ponytail: static defaults matching ui.load_29m(); no config parsing.
        self._configured = dict(
            mode="pretrain", d_model=1600, n_heads=25, n_loops=6,
            steps=1000, batch_size=2, seq_len=64,
            lr_muon=0.0005, lr_adamw=0.0001,
            image_prob=0.5, ckpt_every=50,
        )
        self._logs_widget.write("[tui] loaded 29m defaults")

    def _start_training(self) -> None:
        params = getattr(self, "_configured", None) or dict(
            mode="pretrain", d_model=768, n_heads=12, n_loops=6,
            steps=50, batch_size=2, seq_len=64,
            lr_muon=0.002, lr_adamw=0.0004,
            image_prob=0.5, ckpt_every=25,
        )
        self.ctrl.configure(**params)
        ok = self.ctrl.start(resume=False)
        if not ok:
            self._logs_widget.write("[tui] already running — ignored")
        else:
            self._logs_widget.write(f"[tui] start mode={params['mode']} steps={params['steps']}")

    # -- thinking --------------------------------------------------------

    @work(thread=True)
    def _run_thinking(self) -> None:
        out = self.query_one("#think_out", RichLog)
        meter = self.query_one("#depth_meter", ProgressBar)
        prompt = self.query_one("#prompt_input", Input).value.strip()
        if not prompt:
            out.write("[dim]empty prompt[/dim]")
            return
        try:
            max_len = int(self.query_one("#think_maxlen", Input).value or "128")
            temp = float(self.query_one("#think_temp", Input).value or "0.7")
        except ValueError:
            out.write("[red]invalid max_len / temp[/red]")
            return
        out.write(f"[dim]> {prompt}[/dim]")
        out.write("[dim]loading model (first run may be slow)…[/dim]")
        try:
            from think import build_think_model, generate
            model, loops, device = build_think_model()
            pairs = generate(model, prompt, max_new_bytes=max_len, temp=temp,
                             max_loops=loops, device=device)
            current_line = []
            for b, depth in pairs:
                ratio = max(0.0, min(1.0, depth / max(loops, 1)))
                meter.update(total=100, progress=ratio * 100)
                if not (0 <= b <= 255):
                    ch = f"<{b}>"
                elif b == 10:
                    if current_line:
                        out.write("".join(current_line))
                        current_line = []
                    continue
                else:
                    ch = chr(b)
                color = "green" if ratio < 0.5 else ("yellow" if ratio < 0.8 else "red")
                current_line.append(f"[{color}]{ch}[/{color}]")
            if current_line:
                out.write("".join(current_line))
            meter.update(total=100, progress=0)
        except Exception as e:
            out.write(f"[red]thinking failed: {type(e).__name__}: {e}[/red]")


if __name__ == "__main__":
    AriaTUI().run()
