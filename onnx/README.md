This folder contains scripts for exporting the KModel to ONNX format.

It uses uv for dependency management. You can also run the scripts by manually installing the specified dependencies.

The expectation is that you run the scripts from the project root directory.

With `uv` installed, you can run the scripts with:

```bash
uv run onnx/export.py
```
or without uv:

```bash
python onnx/export.py
```

The `export.py` script will export the model to `kokoro.onnx`.

The `verify.py` script will verify the exported model by comparing the output of the original model with the output of the exported model.

```bash
uv run onnx/verify.py
```

This will output a `torch_output.wav` and a `onnx_output.wav` file in the `onnx` directory, as well as print the MSE of the two audio files. Make sure this is close to 0. You can adjust the exported text by modifying the `text` variable in the `verify.py` script.



