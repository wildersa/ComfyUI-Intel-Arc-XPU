# ComfyUI-Intel-Arc-XPU

Otimizações experimentais para rodar o **ComfyUI no Windows com GPUs Intel Arc (`xpu`)** unindo duas melhorias complementares:

1. **Custom Node `comfy_kitchen` backend `'xpu'` em PyTorch puro** — acelera operações quantizadas `INT8` (`int8_tensorwise` / `convrot`) e reduz drasticamente o pico de VRAM na dequantização sem depender de CUDA/Triton.
2. **Build Windows do `comfy-aimdo-xpu` (Release `v0.1.0`)** — habilita suporte a **DynamicVRAM** no Windows para Intel Arc (`xpu`) via Level Zero + Microsoft Detours hook em cima do `torch==2.13.0+xpu` (`sycl9.dll`).

---

> [!WARNING]
> **AVISO IMPORTANTE / DISCLAIMER (LEIA ANTES DE USAR)**
>
> **Ainda não fizemos testes o suficiente — este conjunto foi validado apenas na minha máquina ("works on my machine").**
>
> - **Hardware & SO testados:** Windows 11, GPU **Intel Arc Pro B70 (`xpu`)**, **Python 3.13 (`cp313`)**, **PyTorch `2.13.0+xpu`** + **`torchvision 0.28.0+xpu`** (`intel-sycl-rt==2026.0.0` / `sycl9.dll`), **ComfyUI `v0.45.3`**.
> - **O que já foi testado e passou:**
>   - `test_xpu_backend.py` (`ALL TESTS PASSED`, incluindo microbenchmark INT8 e redução de pico de VRAM);
>   - Workflow real `VNCCS_PoseStudio_QI2.1` (`qwen_image_2.1_int8_convrot.safetensors`, `1024x1024`, `steps=25`, LoRA ativo) com ganho de **~19%** no sampler (`37s` -> `31s`, `1,51–1,52 s/it` -> `1,20 s/it` steady loop / `1,25 s/it` média);
>   - Boot real do ComfyUI Desktop com log `DynamicVRAM support detected and enabled`;
>   - Testes de fumaça e ciclo de vida do hook Windows/VBAR (`scripts\test-windows-xpu-hook.cmd`, `run_windows_ur_hook_smoke.py`, `run_xpu_vbar_resident_growth.py`, `run_xpu_vbar_retirement_safety.py` default & async 100 ciclos, `run_xpu_vbar_consumer_lifecycle.py`).
> - **Limitações e testes pendentes:**
>   - Dois scripts sintéticos de política de memória do repositório upstream (`vbar-pageout-xpu.py` e `repro_xpu_platform_memory_policy.py`) falharam em suas asserções (`AssertionError: []` e `AssertionError: [1]`);
>   - A suíte `pytest` upstream (`tests/test_xpu_backend.py`) não foi executada nesse `.venv`;
>   - Workflows pesados de vídeo (como `wan_animate`) com **DynamicVRAM + backend `'xpu'`** ainda não foram benchmarkados ponta a ponta.
>
> *English summary: **Experimental / early release.** Tested only on Windows 11 + Intel Arc Pro B70 + `torch==2.13.0+xpu` + ComfyUI `v0.45.3`. Works on my machine, not yet broadly tested across all Intel Arc SKUs or long multi-model workflows.*

---

## O que fizemos e como funciona

### Melhoria 1 — Backend `'xpu'` em PyTorch puro (`xpu_backend.py` + `__init__.py`)

Por padrão, o `comfy_kitchen` prioriza backends CUDA/Triton e cai no fallback `eager` em GPUs Intel Arc (`xpu`). No fallback `eager`, diversas dequantizações promovem pesos para `float32` temporariamente na VRAM e o ComfyUI desabilita caminhos rápidos `int8_tensorwise` em XPU.

O custom node deste repositório registra automaticamente um backend `'xpu'` na inicialização do ComfyUI (`__init__.py` -> `install_comfy_xpu_hooks()`):

- **Registro com prioridade no `comfy_kitchen.registry`:** insere o backend `'xpu'` imediatamente antes de `triton`/`eager`.
- **Dequantização sem explosão para FP32 (`_dequantize_simple_to`, `dequantize_int8_simple`, `dequantize_int8_simple_dtype`):** converte pesos quantizados diretamente para o `dtype` alvo (`bfloat16` ou `float16`), cortando pela metade o pico de VRAM temporária em relação ao `eager`.
- **Un-rotation ConvRot em chunks (`dequantize_int8_convrot_weight`, `dequantize_int8_convrot_weight_dtype`):** processa a etapa FP32 em blocos limitados por `_FP32_CHUNK_ELEMS = 8 << 20` (~32 MB de scratch) antes de retornar para `bf16`/`fp16`.
- **Matmul híbrido `int8_linear`:**
  - Quando `M >= K` (camadas dominadas por tokens) e entrada/saída estão em `fp16`/`bf16`, usa dequantização em baixa precisão + `torch.nn.functional.linear` nos motores XMX da Intel Arc;
  - Quando `M < K`, usa `torch._int_mm(a, weight_nk.T)` sem cópia extra da transposta (com padding automático para múltiplos de 8 quando necessário) e reescala FP32 em chunks acima de `_RESCALE_CHUNK_ELEMS = 16 << 20`.
- **Desbloqueio de quantização nativa no ComfyUI (`install_comfy_xpu_hooks`):** aplica patch idempotente em:
  - `comfy.model_management.supports_int8_compute`;
  - `comfy.ops.get_disabled_quant_formats` (habilita `int8_tensorwise` no XPU, mantendo `convrot_w4a4` e `_GROUPED_INT8_FORMATS` desabilitados);
  - `TensorWiseINT8Layout.supports_fast_matmul`.

#### Resultados medidos (Melhoria 1 isolada)

| Teste | Baseline (`eager` / `torch==2.12.1+xpu`) | Com backend `'xpu'` | Ganho |
| --- | --- | --- | --- |
| `test_xpu_backend.py` (`M=1024, K=5120, N=13824`) | `5.69 ms` / pico `513.0 MB` | `1.58 ms` / pico `123.0 MB` | **~3.6x** mais rápido, **-76%** pico VRAM |
| Dequant peak VRAM (`test_xpu_backend.py`) | `540.0 MB` | `272.0 MB` | **~50%** menos VRAM temporária |
| Workflow `VNCCS_PoseStudio_QI2.1` (`1024x1024`, `25` steps, LoRA) | `1,51–1,52 s/it` (`37s` sampler) | `1,20 s/it` steady (`1,25 s/it` média, `31s` sampler) | **~19%** mais rápido |

---

### Melhoria 2 — `comfy-aimdo-xpu` compilado para Windows (`DynamicVRAM`)

O `comfy-aimdo` oficial traz suporte de DynamicVRAM focado em CUDA, e no Windows não existe pacote pré-compilado oficial para Intel Arc (`xpu`). Para habilitar alocação virtual VBAR e eviction inteligente de modelos em GPUs Intel Arc no Windows:

1. Clonamos o fork [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu) no commit exato `3bc6afbcb92e10a2d4737e79d170190ed787bf89` (mesmo commit referenciado pelo projeto oficial [`intel/llm-scaler`](https://github.com/intel/llm-scaler)), junto de `oneapi-src/level-zero` e `microsoft/Detours`.
2. Compilamos o **Microsoft Detours** (`scripts\build-windows-detours.cmd` -> `detours.lib`).
3. Ajustamos o script `scripts\build-windows-xpu.cmd` (linha 134) adicionando a biblioteca **`dxcore.lib`**, necessária no Windows SDK `10.0.26100.0` para resolver `DXCoreCreateAdapterFactory` usado em `src-win\shmem-detect.c`.
4. Compilamos a `aimdo_xpu.dll` e empacotamos o wheel Windows (`abi3`):
   - **Wheel (Release `v0.1.0`):** `comfy_aimdo-0.6.0+xpu.g3bc6afb.dirty-cp39-abi3-win_amd64.whl` (`157.964` bytes)
   - **SHA-256 do wheel:** `68dde78cae9655a2ced31f2595e43b7e0026341ac6d6e7172226a5377820aaae`
   - **SHA-256 da `aimdo_xpu.dll` embutida:** `B1850393799D9F1D3E70A47754564737173F0ADEF4A699CA3B41F27E5ABA1CA3`

> [!IMPORTANT]
> **Requisito obrigatório: `torch==2.13.0+xpu` (`sycl9.dll`)**
>
> A `aimdo_xpu.dll` compilada com o Intel oneAPI 2026.0 depende do runtime **`sycl9.dll`** (`intel-sycl-rt==2026.0.0`).
> Se você ainda estiver no `torch==2.12.1+xpu` (`sycl8.dll`), o carregamento da extensão XPU irá falhar até atualizar o PyTorch XPU para `2.13.0+xpu`.

Além disso, o `main.py` do ComfyUI só ativa DynamicVRAM automaticamente em GPUs NVIDIA/ROCm (`dynamic_vram_supported()`), portanto em Intel Arc você **precisa passar explicitamente** `--enable-dynamic-vram` na inicialização (recomendamos também `--reserve-vram 4`).

---

## Como instalar

### 1. Instalar o Custom Node (`comfy_kitchen` backend `'xpu'`)

No diretório do seu ComfyUI (`<ComfyUI_Root>`), clone este repositório dentro de `custom_nodes`:

```powershell
cd "<ComfyUI_Root>\custom_nodes"
git clone https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
```

### 2. Atualizar o `.venv` do ComfyUI para `torch==2.13.0+xpu`

Faça backup do seu ambiente atual antes de alterar pacotes:

```powershell
"<ComfyUI_Root>\.venv\Scripts\python.exe" -m pip freeze > venv_backup_freeze.txt
"<ComfyUI_Root>\.venv\Scripts\python.exe" -m pip install --upgrade torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

*(Ajuste `torchaudio` ou outros pacotes se o seu ambiente utilizar versões específicas.)*

### 3. Instalar o wheel `comfy-aimdo-xpu` da Release `v0.1.0`

Baixe `comfy_aimdo-0.6.0+xpu.g3bc6afb.dirty-cp39-abi3-win_amd64.whl` na página de [Releases](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases) e instale no `.venv` do ComfyUI:

```powershell
"<ComfyUI_Root>\.venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps "<Caminho_Download>\comfy_aimdo-0.6.0+xpu.g3bc6afb.dirty-cp39-abi3-win_amd64.whl"
"<ComfyUI_Root>\.venv\Scripts\python.exe" -m pip check
```

### 4. Configurar as flags de inicialização do ComfyUI

Adicione as flags abaixo aos argumentos de lançamento do ComfyUI (ou no `launchArgs` do ComfyUI Desktop):

```text
--enable-dynamic-vram --reserve-vram 4
```

Ao iniciar o ComfyUI, confira no log (`comfyui.log`) se aparece:

- `DynamicVRAM support detected and enabled`
- `Native ops: int8_tensorwise` / registro do backend `'xpu'` no `comfy_kitchen`

### 5. Rodar o autoteste do backend `'xpu'` (opcional)

```powershell
"<ComfyUI_Root>\.venv\Scripts\python.exe" "<ComfyUI_Root>\custom_nodes\ComfyUI-Intel-Arc-XPU\test_xpu_backend.py"
```

---

## Como compilar o `comfy-aimdo-xpu` do zero (Windows)

Para auditoria e reprodutibilidade sob a licença GPL-3.0:

1. Pré-requisitos instalados:
   - **Visual Studio Build Tools / MSVC (`cl.exe`)** + **Windows SDK** (`10.0.26100.0` testado)
   - **Intel oneAPI Base Toolkit 2026.0** (`dpcpp` / `icx` / `sycl9` / `libur_win`)
   - **Python 3.13** com `setuptools` e `wheel`
2. Clonar as fontes:
   - `xiangyuT/comfy-aimdo-xpu` em `3bc6afbcb92e10a2d4737e79d170190ed787bf89`
   - `oneapi-src/level-zero` em `build\level-zero-src`
   - `microsoft/Detours` em `build\detours-src`
3. Compilar Detours via `scripts\build-windows-detours.cmd`.
4. Editar `scripts\build-windows-xpu.cmd` na chamada do linker para incluir `dxcore.lib` junto às demais `.lib` do Windows SDK.
5. Executar `scripts\build-windows-xpu.cmd`, validar com `scripts\test-windows-xpu-hook.cmd` e empacotar com `python -m pip wheel --no-build-isolation --no-deps . -w dist`.

---

## Estrutura do repositório

- [`__init__.py`](./__init__.py) — Ponto de entrada do custom node no ComfyUI; chama `install_comfy_xpu_hooks()`.
- [`xpu_backend.py`](./xpu_backend.py) — Implementação do backend `'xpu'` para `comfy_kitchen` + hooks do ComfyUI.
- [`test_xpu_backend.py`](./test_xpu_backend.py) — Testes unitários de corretude e microbenchmark de latência/VRAM.
- [`LICENSE`](./LICENSE) — Licença **GNU General Public License v3.0 (GPL-3.0)**.
- **Binário Windows (`.whl`):** publicado exclusivamente em [Releases (`v0.1.0`)](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases) para manter o clone de `custom_nodes` limpo.

---

## Créditos e Licença

- Distribuído sob a licença **GPL-3.0** ([`LICENSE`](./LICENSE)), em conformidade com o `comfy-aimdo` / `comfy-aimdo-xpu`.
- Créditos dos projetos upstream:
  - [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu) (commit `3bc6afbcb92e10a2d4737e79d170190ed787bf89`)
  - [`intel/llm-scaler`](https://github.com/intel/llm-scaler)
  - [`Comfy-Org/comfy-aimdo`](https://github.com/Comfy-Org/comfy-aimdo)
  - [`oneapi-src/level-zero`](https://github.com/oneapi-src/level-zero)
  - [`microsoft/Detours`](https://github.com/microsoft/Detours)
