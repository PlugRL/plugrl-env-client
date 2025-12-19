# 🚀 plugrl-worker

**plugrl-worker** (Vision Language Action Reinforcement Learning Infrastructure) is a foundational framework for building and running distributed reinforcement learning agents. It's designed to streamline the communication between environment workers and a centralized training server, enabling efficient and scalable RL experiments.

## ✨ Features

  * **Distributed Communication**: Utilizes `websockets` and `msgpack` for fast, secure, and asynchronous data transfer between the central server and multiple environment workers.
  * **Modular Design**: Separates the core infrastructure (`plugrl-worker`) from the client-side components (`plugrl-client`), allowing for independent development and deployment.
  * **Command-Line Interface**: Provides a user-friendly CLI powered by `tyro` for starting and managing RL tasks.
  * **Gymnasium Integration**: Seamlessly works with `Gymnasium` environments for standardized and flexible environment interaction.

-----

## 🛠️ Installation

The easiest way to get started is by cloning the repository and using Poetry to handle the dependencies.

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:CTP314/plugrl-worker.git
    cd plugrl-worker
    ```

2.  **Install dependencies with Pip:**

    ```bash
    pip install -e .
    ```

3. **Install Optional Environment Dependencies**
   
    ```
    pip install -e ".[robomimic, atari, classic]"
    ```

## 🚀 Usage

The `plugrl-run-worker` tool launches worker processes for specific environments.

### Basic Syntax

```bash
poetry run plugrl-run-worker <ENVIRONMENT_TYPE> [OPTIONS]
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

2.  **Run the Worker with the Flag:**
    In your `plugrl-server` project, execute the worker command, making sure to include the `--use-remote-viewer` flag:

    ```bash
    # Example: Run the dummy worker and stream data to the viewer on port 8001
    plugrl-run-worker dummy-v1 --use-remote-viewer
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
poetry run plugrl-run-worker dummy-v1 --num-episodes 100 --log-level debug

# Run with custom environment settings (64x64 image, 4-dim action space)
poetry run plugrl-run-worker dummy-v1 --env.img-width 64 --env.img-height 64 --env.action-dim 4
```

### Get More Help

To see all available options for a specific environment:

```bash
poetry run plugrl-run-worker dummy-v1 --help
```