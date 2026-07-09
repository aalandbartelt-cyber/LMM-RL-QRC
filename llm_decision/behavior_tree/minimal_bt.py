"""Minimal behavior-tree placeholder for task JSON execution."""


def run_task(task):
    for step in task.get("steps", []):
        print(f"execute: {step.get('action')} target={step.get('target', '')}")
    return {"task_status": "success"}


if __name__ == "__main__":
    demo_task = {
        "task_id": "demo",
        "steps": [
            {"action": "go_to", "target": "p1"},
            {"action": "inspect", "target": "sign_1"},
            {"action": "return_home"},
        ],
    }
    print(run_task(demo_task))
