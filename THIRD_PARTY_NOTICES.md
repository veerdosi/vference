# Third-party notices

The native stable-buffer design in `native/extension.cpp` was informed by the
Apache-2.0-licensed owned expert-pool implementation in
[AMOS144/Vates](https://github.com/AMOS144/Vates), audited at commit
`7826e41821c282464cc10c80a4af4414023720e3`. vference's implementation is
adapted to its own single-pack format and intentionally excludes Vates'
prefetch, speculative decoding, KV quantization, and model-specific policies.

Vates copyright and license terms are available in its repository:
<https://github.com/AMOS144/Vates/blob/7826e41821c282464cc10c80a4af4414023720e3/LICENSE>.
