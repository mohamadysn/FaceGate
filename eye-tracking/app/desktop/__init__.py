"""Desktop GUI for face recognition (enrollment, live ID, gallery)."""

__all__ = ["run_app"]


def run_app() -> None:
    from .app import FaceRecogApp

    FaceRecogApp().run()
