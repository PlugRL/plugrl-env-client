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
    uv sync --extra robomimic --extra atari --extra classic --extra libero
    ```

    ```
    pip install -e ".[robomimic, atari, classic, libero]"
    ```

    Note: `robomimic` and `libero` pull in `egl-probe`, whose legacy CMake build
    needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` when using CMake 4+. `uv` is
    configured in this repository to apply that automatically. If you install
    with `pip`, set the variable manually, e.g.

    ```bash
    CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e ".[libero]"
    ```

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
