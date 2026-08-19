# vs-remote

Remote execution server and frame proxy for VapourSynth.

`vs-remote` runs VapourSynth scripts on a remote machine (e.g. headless server or workstation) and streams frames to a local client for previewing or encoding.

It mirrors remote clips as local `VideoNode` proxies, streaming frames on demand with asynchronous prefetching, uses ZeroMQ for communication and supports multiple independent outputs.

## Limitations

- AudioNode outputs are not supported.
- Variable resolution and format clips are not supported.
- Frame property serialization only preserves primitive types (`int`, `float`, `str`, `bytes`, and lists of primitives).

    Non-primitive objects such as embedded `VideoFrame` references (`_Alpha`) fall back to their string `repr()`.

---

## Installation

```bash
uv add vsremote
```

Or with `pip`:

```bash
pip install vsremote
```

---

## Quick Start

### 1. Start the Remote Server

On the machine hosting VapourSynth and source media:

```bash
vsremote serve path/to/script.vpy --address tcp://127.0.0.1:5555
```

Or programmatically in Python:

```python
import vsremote

vsremote.serve("script.vpy", address="tcp://127.0.0.1:5555")
```

---

### 2. Connect from the Client

On the local machine:

```python
# client_script.vpy
import vapoursynth as vs
import vsremote

# Mirror output 0 from the remote server
clip = vsremote.source("tcp://192.168.1.100:5555", output=0)

clip.set_output()
```

Open `client_script.vpy` in `vsview`, or pipe directly via the CLI:

```bash
# Preview in vsview
vsview client_script.vpy

# Pipe directly from CLI
vsremote pipe tcp://192.168.1.100:5555 --output 0 | ffmpeg -i - -c:v libx264 out.mp4
```

---

## CLI Reference

| Command  | Description                                                         | Example                                                                               |
| :------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------------------ |
| `serve`  | Host a `.vpy` script or execution server                            | `vsremote serve script.vpy --address tcp://127.0.0.1:5555`                            |
| `ping`   | Test connection and measure round-trip latency                      | `vsremote ping tcp://192.168.1.100:5555`                                              |
| `info`   | Display metadata for all outputs on the remote server               | `vsremote info tcp://192.168.1.100:5555`                                              |
| `pipe`   | Stream frames directly to stdout as Y4M or raw planes               | `vsremote pipe tcp://192.168.1.100:5555 --y4m --output 0 \| x265 --y4m - -o out.hevc` |
| `keygen` | Generate a Curve25519 keypair for CurveZMQ encryption & client auth | `vsremote keygen`                                                                     |

---

## Python API Reference

### Client API

#### `vsremote.source(...)`

Create a local `vs.VideoNode` proxy mirroring a remote output:

```python
import vsremote

clip = vsremote.source(
    address="tcp://192.168.1.100:5555",
    output=0,
    compression="zstd",  # "zstd" or "none"
    prefetch=4,  # Frames to asynchronously prefetch ahead
    auth_token=None,  # Optional authentication token
    curve_server_key=None,  # Optional CurveZMQ server public key
    curve_public_key=None,  # Optional CurveZMQ client public key
    curve_secret_key=None,  # Optional CurveZMQ client secret key
    forward_logs=True,  # Stream remote logs to local logging
)
```

#### `vsremote.RemoteClient`

Client for output introspection, dynamic script loading, and multi-output retrieval:

```python
import asyncio

import vsremote

with vsremote.RemoteClient("tcp://192.168.1.100:5555") as client:
    # Introspect available outputs
    outputs = client.list_outputs().result()
    for item in outputs:
        print(f"[{item.index}] {item.name}: {item.info.width}x{item.info.height}")

    # Get proxies for specific or all clips
    clip0 = client.get_output(0)
    all_clips = client.get_outputs()  # dict[int, vs.VideoNode]


async def main() -> None:
    # Also usable as asynchronous context manager
    async with vsremote.RemoteClient("tcp://192.168.1.100:5555") as client:
        # Dynamic control (requires server started with --allow-eval)
        await client.reload()  # Reload script from disk
        await client.load_script("/path/to/another.vpy")  # Switch active script
        await client.load_code("import vapoursynth as vs; vs.core.std.BlankClip().set_output()")


asyncio.run(main())
```

---

### Remote Script Authoring API

When authoring `.vpy` scripts served by `vsremote`, use `vsremote.set_output` to register named outputs:

```python
# server_script.vpy
import vapoursynth as vs

from vsremote import is_preview, set_output

core = vs.core

src = core.bs.VideoSource("source.mkv")
noartifact = core.noise.Add(src, var=2000)

set_output(src)
set_output(noartifact)

if is_preview():
    print("Running inside vsremote server environment")
```

---

## Security

VapourSynth scripts are Python code. Evaluating untrusted `.vpy` scripts or enabling `--allow-eval` grants arbitrary code execution privileges within the server process.

- **Defaults**: The server binds to `127.0.0.1` with `--allow-eval` disabled by default.
- **Remote / WAN (Recommended)**: Use SSH port forwarding so no ZeroMQ ports are exposed to the internet:

    ```bash
    # Remote server (bind to localhost)
    vsremote serve script.vpy --address tcp://127.0.0.1:5555

    # Local client (SSH tunnel)
    ssh -N -L 5555:127.0.0.1:5555 user@remote-server.com

    # Connect locally
    vsremote info --address tcp://127.0.0.1:5555
    ```

- **Direct LAN (Encryption Only)**: Enable CurveZMQ (Curve25519) encryption and optional pre-shared token authentication:

    ```bash
    # Server (generate ephemeral keypair or provide static secret key)
    vsremote serve script.vpy --address tcp://192.168.1.100:5555 --curve-secret-key "<SERVER_SECRET>" --auth-token "secret"

    # Client
    vsremote info tcp://192.168.1.100:5555 --curve-server-key "<SERVER_PUBLIC>" --auth-token "secret"
    ```

- **Direct LAN (Mutual Authentication & Whitelisting)**: Whitelist authorized client public keys on the server:

    ```bash
    # Server (whitelist allowed client public keys)
    vsremote serve script.vpy --address tcp://192.168.1.100:5555 --curve-secret-key "<SERVER_SECRET>" --curve-allowed-keys "<CLIENT_PUBLIC>"

    # Client (connect with client keypair)
    vsremote info tcp://192.168.1.100:5555 --curve-server-key "<SERVER_PUBLIC>" --curve-public-key "<CLIENT_PUBLIC>" --curve-secret-key "<CLIENT_SECRET>"
    ```

- **Untrusted Scripts/Code**: Run `vsremote` in a rootless container with read-only mounts and dropped capabilities.

---

## Architecture

```mermaid
flowchart BT
    subgraph Client ["Client Machine"]
        SRC["vsremote.source / RemoteClient"]
        CT["ClientTransport (DEALER)"]
        SUB["Stream Subscriber (SUB)"]

        SRC -->|"ModifyFrame"| CT
    end

    ZMQ{{"ZeroMQ Transport\n(TCP / IPC)"}}

    subgraph Server ["Rendering Server"]
        SD["ServerDaemon (ROUTER)"]
        VSE["VapourSynth Core / vsengine"]

        SD -->|"get_frame_async"| VSE
        VSE -->|"vs.VideoFrame"| SD
    end

    CT <-->|"Requests & Frames"| ZMQ
    ZMQ <-->|"Render Requests"| SD
    SD -->|"PUB Logs & Output"| ZMQ
    ZMQ -->|"Events"| SUB
```

---

## Notes

This project was developed with the assistance of AI coding tools.
