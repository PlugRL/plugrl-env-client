# 🚀 plugrl-env-client

**plugrl-env-client** (Vision Language Action Reinforcement Learning Infrastructure) runs environments and talks to a centralized training server, enabling scalable RL experiments.

## ✨ Features

  * **Distributed Communication**: Utilizes `websockets` and `msgpack` for fast, secure, and asynchronous data transfer between the central server and multiple environment workers.
    * **Modular Design**: Separates the environment-side runtime (`plugrl-env-client`) from the shared protocol layer (`plugrl-protocol`).
  * **Command-Line Interface**: Provides a user-friendly CLI powered by `tyro` for starting and managing RL tasks.
  * **Gymnasium Integration**: Seamlessly works with `Gymnasium` environments for standardized and flexible environment interaction.

-----

## 🛠️ Installation

The easiest way to get started is by cloning the repository and using `uv` to manage the environment.

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:PlugRL/plugrl-env-client.git
    cd plugrl-env-client
    ```

2.  **Install dependencies with Pip:**

    **Option A: Use uv (recommended)**

    ```bash
    uv sync
    ```

    **Option B: Editable install with pip**

    ```bash
    pip install -e .
    ```

3. **Install Optional Environment Dependencies**

    With `uv`:

    ```bash
    uv sync --extra robomimic --extra atari --extra classic
    ```

    ```
    pip install -e ".[robomimic, atari, classic]"
    ```

### Manual RoboCasa Setup

`Robocasa-v1` is intentionally documented as a manual install instead of a built-in extra because RoboCasa requires a specific editable-install flow, external assets, and local setup scripts.

Follow the upstream RoboCasa instructions in [`/mnt/robocasa/README.md`](/mnt/robocasa/README.md). The validated plan for `plugrl-env-client` is:

1. Use Python `3.11`.
2. Install `plugrl-env-client` itself in that environment.
3. Clone and editable-install `robosuite` from the upstream `master` branch:

   ```bash
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```

4. Editable-install RoboCasa:

   ```bash
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   ```

5. Run RoboCasa setup scripts:

   ```bash
   python -m robocasa.scripts.setup_macros
   python -m robocasa.scripts.download_kitchen_assets
   ```

Notes:

- RoboCasa's README explicitly recommends Python `3.11`.
- RoboCasa's gym environments are registered through `robocasa.wrappers.gym_wrapper`, so a plain source checkout without proper editable install is not enough.
- Asset download is required before real environment rollouts.
- I did not run these installation steps automatically here; this section is the handoff plan for your manual setup.

## 🚀 Usage

The `plugrl-run-env-client` tool launches one or more env client processes for specific environments.

> Note: `plugrl-run-worker` is kept as a backwards-compatible alias.

### Basic Syntax

```bash
uv run plugrl-run-env-client <ENVIRONMENT_TYPE> [OPTIONS]
```

### Remote Viewer Usage (`--use-remote-viewer`)

The `--use-remote-viewer` flag enables the worker to send environment observation data (like images and states) to a separate, remote viewer application for real-time visualization.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--use-remote-viewer` | FLAG | `False` | **Enables** the worker to stream data to a remote viewer. |
| `--viewer-host` | STR | `0.0.0.0` | IP address where the viewer is expected to be running. |
| `--viewer-port` | INT | `8001` | Port where the viewer is expected to be listening for connections. |

#### How to Use the Remote Viewer

To utilize this feature, you must first start the **viewer frontend** in a separate terminal, typically from the **`plugrl-viewer`** project.

1.  **Start the Viewer Frontend:**
    In your `plugrl-viewer` project directory, run the following command to start the web application (listening on the default port `8001`):

    ```bash
    uvicorn plugrl_viewer.main:app --reload --port 8001
    ```

2.  **Run the Env Client with the Flag:**
    In your `plugrl-server` project, execute the worker command, making sure to include the `--use-remote-viewer` flag:

    ```bash
    # Example: Run the dummy worker and stream data to the viewer on port 8001
    plugrl-run-env-client dummy-v1 --use-remote-viewer
    ```

The worker will then attempt to establish a WebSocket connection with the viewer at the specified host and port (defaulting to `0.0.0.0:8001`) and begin streaming environment data.

### Available Environments

| Type | Description |
| :--- | :--- |
| **dummy-v1** | Dummy/testing environment |
| **classic-v1** | Classic control environments (e.g., CartPole) |
| **robomimic-v1** | RoboMimic-based robotic manipulation environment |
| **robocasa-v1** | RoboCasa kitchen manipulation environment with 3-view RGB + state + prompt |
| **atari-v1** | Atari game environment |

### Examples

Run the `dummy-v1` environment with custom parameters:

```bash
# Run 100 episodes with 'debug' logging level
uv run plugrl-run-env-client dummy-v1 --num-episodes 100 --log-level debug

# Run with custom environment settings (64x64 image, 4-dim action space)
uv run plugrl-run-env-client dummy-v1 --env.img-width 64 --env.img-height 64 --env.action-dim 4
```

### Get More Help

To see all available options for a specific environment:

```bash
uv run plugrl-run-env-client dummy-v1 --help
```

### RoboCasa Examples

After you have manually completed the RoboCasa installation steps above, you can inspect the RoboCasa-specific CLI options with:

```bash
uv run plugrl-run-env-client robocasa-v1 --help
```

A typical rollout command against a plugrl-compatible policy server looks like:

```bash
uv run plugrl-run-env-client robocasa-v1 \
  --num-envs 8 \
  --num-episodes 8 \
  --env.task-name SearingMeat \
  --env.split target \
  --env.action-encoding passthrough \
  --recorder.record-video \
  --recorder.record-full-rollout \
  --recorder.record-debug-packets \
  --recorder.episode-freq 1
```

This setup records:

- three per-view full-rollout mp4 files under `runs/.../rollout/proc_000/videos/full/images/`
- first infer / infer-response / feedback debug packets under `runs/.../rollout/proc_000/debug_packets/`
- sampled observation artifacts when `episode-freq` is enabled

### Manual RoboCasa Tests

The repository now includes manual RoboCasa pytest cases under [`tests/test_robocasa_manual.py`](/mnt/plugrl/plugrl-env-client/tests/test_robocasa_manual.py). They are intentionally not part of the default automated suite.

After you finish the manual RoboCasa installation, you can run:

```bash
pytest -m manual tests/test_robocasa_manual.py::test_manual_robocasa_parallel_rollout_smoke -s
```

To validate real OpenPI server interoperability, first export:

```bash
export OPENPI_ROBOCASA_CONFIG_NAME=...
export OPENPI_ROBOCASA_CHECKPOINT_DIR=...
export OPENPI_ROBOCASA_DATASET_DIR=...
# optional override, defaults to /mnt/openpi-base/.venv/bin/python
export OPENPI_ROBOCASA_SERVER_PYTHON=...
```

Then run:

```bash
pytest -m manual tests/test_robocasa_manual.py::test_manual_robocasa_openpi_server_roundtrip -s
```
