#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 一键部署脚本：在远程云服务器上无 Docker 构建 RL 项目环境
# ============================================================================
# 用法：
#   bash ./deploy.sh
#
# 可选环境变量：
#   CONDA_DIR                  Miniconda 安装路径（默认 /opt/conda）
#   CONDA_ENV                  conda 环境名（默认 rl）
#   PYTHON_VERSION             Python 版本（默认 3.8.10）
#   TORCH_VERSION              PyTorch 版本（默认 2.4.1）
#   TORCHVISION_VERSION        torchvision 版本（默认 0.19.1）
#   CUDA_VERSION               CUDA 版本（默认 12.1，用于 conda pytorch-cuda）
#   USE_CN_MIRROR              是否使用国内镜像加速（默认 1）
#   SKIP_SYSTEM_DEPS           跳过系统依赖安装（默认 0）
#   SKIP_CONDA                 跳过 Miniconda 安装（默认 0）
#   SKIP_PIP                   跳过 pip 依赖安装（默认 0）
#   SKIP_ISAAC                 跳过 Isaac Gym/Envs 安装（默认 0）
#   SKIP_MUJOCO                跳过 MuJoCo 安装（默认 0）
#   SKIP_VNC                   跳过 VNC 安装（默认 0）
#   SKIP_LEGGED                跳过 legged_gym/rsl_rl 安装（默认 0）
#   NO_GPU_VERIFY              跳过GPU验证但仍安装CUDA依赖（默认 0）
#   MUJOCO_DIR                 MuJoCo 安装目录（默认 ~/.mujoco）
#   ISAACGYMENVS_GIT_URL       IsaacGymEnvs 仓库地址
#   ISAACGYM_ARCHIVE           Isaac Gym 压缩包路径（默认自动搜索）
#   PROJECT_DIR                项目根目录（默认脚本所在目录）
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
CONDA_DIR="${CONDA_DIR:-/opt/conda}"
CONDA_ENV="${CONDA_ENV:-himloco}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8.10}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
USE_CN_MIRROR="${USE_CN_MIRROR:-1}"
SKIP_SYSTEM_DEPS="${SKIP_SYSTEM_DEPS:-0}"
SKIP_CONDA="${SKIP_CONDA:-0}"
SKIP_PIP="${SKIP_PIP:-0}"
SKIP_ISAAC="${SKIP_ISAAC:-0}"
SKIP_MUJOCO="${SKIP_MUJOCO:-0}"
SKIP_VNC="${SKIP_VNC:-0}"
SKIP_LEGGED="${SKIP_LEGGED:-0}"
NO_GPU_VERIFY="${NO_GPU_VERIFY:-1}"
MUJOCO_DIR="${MUJOCO_DIR:-${HOME}/.mujoco}"
ISAACGYMENVS_GIT_URL="${ISAACGYMENVS_GIT_URL:-https://github.com/isaac-sim/IsaacGymEnvs.git}"
ISAACGYM_ARCHIVE="${ISAACGYM_ARCHIVE:-}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $(date '+%H:%M:%S') $*"; }

# 带重试的下载函数
download_with_fallback() {
    local out="$1"; shift
    for url in "$@"; do
        [ -n "$url" ] || continue
        log_info "下载尝试: ${url}"
        if curl -fL --retry 20 --retry-delay 3 --retry-all-errors \
            --connect-timeout 20 --max-time 1800 -o "${out}" "${url}"; then
            if [ -s "${out}" ]; then
                log_info "下载成功: ${url}"
                return 0
            fi
        fi
        log_warn "下载失败: ${url}"
    done
    return 1
}

# 带 fallback 的 pip install
pip_install_with_fallback() {
    local requirement="$1"
    log_info "pip 安装: ${requirement}"
    if ${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip install --no-cache-dir --prefer-binary \
        --retries 20 --default-timeout 120 "${requirement}"; then
        return 0
    fi
    log_warn "首次安装失败，回退到官方 PyPI: ${requirement}"
    ${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip install --no-cache-dir --prefer-binary \
        --retries 20 --default-timeout 120 \
        -i https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        "${requirement}"
}

install_vnc() {
    log_step "----- 安装 TurboVNC 和 VirtualGL -----"
    
    local VGL_DIR="/opt/VirtualGL"
    local TURBO_DIR="/opt/TurboVNC"
    local VGL_VER="3.1.4"
    local TURBO_VER="3.3"
    
    mkdir -p /tmp/vnc_inst && cd /tmp/vnc_inst
    
    if [[ ! -d "${VGL_DIR}" ]]; then
        log_info "下载 VirtualGL ${VGL_VER}..."
        local VGL_TAR="virtualgl_${VGL_VER}_amd64.deb"
        if download_with_fallback "${VGL_TAR}" \
            "https://github.com/VirtualGL/virtualgl/releases/download/${VGL_VER}/${VGL_TAR}" \
            "https://ghproxy.com/https://github.com/VirtualGL/virtualgl/releases/download/${VGL_VER}/${VGL_TAR}"; then
            sudo dpkg -i "${VGL_TAR}" || sudo apt-get install -f -y
            rm -f "${VGL_TAR}"
            log_info "VirtualGL 安装完成"
        fi
    else
        log_info "VirtualGL 已安装"
    fi
    
    if [[ ! -d "${TURBO_DIR}" ]]; then
        log_info "下载 TurboVNC ${TURBO_VER}..."
        local TURBO_DEB="turbovnc_${TURBO_VER}_amd64.deb"
        if download_with_fallback "${TURBO_DEB}" \
            "https://github.com/TurboVNC/turbovnc/releases/download/${TURBO_VER}/${TURBO_DEB}" \
            "https://ghproxy.com/https://github.com/TurboVNC/turbovnc/releases/download/${TURBO_VER}/${TURBO_DEB}"; then
            sudo dpkg -i "${TURBO_DEB}" || sudo apt-get install -f -y
            rm -f "${TURBO_DEB}"
            log_info "TurboVNC 安装完成"
        fi
    else
        log_info "TurboVNC 已安装"
    fi
    
    # 安装 X11 依赖和轻量桌面环境 xfce4（VNC 需要桌面环境才能启动）
    sudo apt-get install -y --no-install-recommends \
        xorg x11vnc xvfb xauth libxtst6 libxcursor1 libxinerama1
    sudo apt-get install -y --no-install-recommends xfce4 xfce4-goodies

    [[ -d "${TURBO_DIR}" ]] && {
        sudo ln -sf "${TURBO_DIR}/bin/vncserver" /usr/local/bin/vncserver
        sudo ln -sf "${TURBO_DIR}/bin/vncstop" /usr/local/bin/vncstop
        sudo ln -sf "${TURBO_DIR}/bin/x0vncserver" /usr/local/bin/x0vncserver
    }
    
    cd "${PROJECT_DIR}" && rm -rf /tmp/vnc_inst
    log_info "VNC 安装完成"
}

# ============================================================================
# 0. 预检
# ============================================================================
log_step "===== 0. 预检 ====="

if [[ $EUID -eq 0 ]]; then
    log_warn "检测到以 root 运行。部分步骤需要 root 权限，将以 sudo 方式执行。"
    HAS_SUDO="1"
else
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        HAS_SUDO="1"
    else
        HAS_SUDO="0"
        log_warn "没有 sudo 权限。系统依赖安装步骤将被跳过。"
        SKIP_SYSTEM_DEPS="1"
    fi
fi

require_cmd() {
    local cmd="$1"
    local hint="$2"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        log_error "缺少命令: ${cmd}"
        log_error "        ${hint}"
        exit 1
    fi
}

require_cmd curl "请先安装 curl"
require_cmd git  "请先安装 git"
require_cmd tar  "请先安装 tar"

if [[ "${SKIP_SYSTEM_DEPS}" != "1" ]] && [[ "${HAS_SUDO}" == "1" ]]; then
    require_cmd apt-get "此脚本仅支持 Ubuntu/Debian 系操作系统"
fi

log_info "项目目录: ${PROJECT_DIR}"
log_info "conda 目录: ${CONDA_DIR}"
log_info "conda 环境: ${CONDA_ENV}"
log_info "Python 版本: ${PYTHON_VERSION}"
log_info "PyTorch 版本: ${TORCH_VERSION}"
log_info "CUDA 版本: ${CUDA_VERSION}"
log_info "使用国内镜像: ${USE_CN_MIRROR}"
log_info "跳过GPU验证: ${NO_GPU_VERIFY}"

# 检测 GPU 可用性
HAS_GPU="0"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q .; then
        HAS_GPU="1"
        log_info "检测到 NVIDIA GPU"
    fi
fi

if [[ "${HAS_GPU}" == "0" ]] && [[ "${NO_GPU_VERIFY}" != "1" ]]; then
    log_warn "未检测到 GPU，但脚本仍会安装 CUDA 版本的 PyTorch"
    log_warn "如需跳过 GPU 相关验证，请设置 NO_GPU_VERIFY=1"
    log_warn "当前将继续执行，但 Isaac Gym 等 GPU 依赖可能无法验证"
fi

# ============================================================================
# 1. 安装系统依赖
# ============================================================================
if [[ "${SKIP_SYSTEM_DEPS}" != "1" ]] && [[ "${HAS_SUDO}" == "1" ]]; then
    log_step "===== 1. 安装系统依赖 ====="
    
    sudo apt-get update
    
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-dev \
        bzip2 \
        git \
        wget \
        ca-certificates \
        patchelf \
        libgl1 \
        libglu1-mesa \
        libosmesa6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libglew2.2 \
        libglvnd0 \
        libglx-mesa0

    log_info "系统依赖安装完成"

    # 安装 libpython3.8（Ubuntu 22.04 需要从 Ubuntu 20.04 源补充）
    log_step "----- 补充 libpython3.8（Isaac Gym gym_38.so 依赖）-----"
    if ! ldconfig -p | grep -q libpython3.8; then
        # 首先尝试通过 apt 安装
        log_info "尝试通过 apt 安装 libpython3.8..."
        if sudo apt-get install -y libpython3.8 2>/dev/null; then
            log_info "libpython3.8 通过 apt 安装成功"
        else
            # 如果 apt 失败，尝试下载 .deb 包
            log_warn "apt 安装失败，尝试手动下载 libpython3.8..."
            mkdir -p /tmp/py38_dep
            cd /tmp/py38_dep
            if download_with_fallback libpython3.8.deb \
                "http://mirrors.aliyun.com/ubuntu/pool/main/p/python3.8/libpython3.8_3.8.10-0ubuntu1~20.04.9_amd64.deb" \
                "http://archive.ubuntu.com/ubuntu/pool/main/p/python3.8/libpython3.8_3.8.10-0ubuntu1~20.04.9_amd64.deb" \
                "http://security.ubuntu.com/ubuntu/pool/main/p/python3.8/libpython3.8_3.8.10-0ubuntu1~20.04.9_amd64.deb"; then
                sudo dpkg -i libpython3.8.deb || true
                sudo apt-get install -f -y || true
            else
                log_warn "libpython3.8 下载失败，但将继续执行（可能影响 Isaac Gym 运行）"
            fi
            rm -rf /tmp/py38_dep
            cd "${PROJECT_DIR}"
        fi
        log_info "libpython3.8 安装完成"
    else
        log_info "libpython3.8 已存在，跳过"
    fi
else
    log_step "===== 1. 跳过系统依赖安装 ====="
fi

# ============================================================================
# 1.5. 安装 VNC（可选）
# ============================================================================
if [[ "${SKIP_VNC}" != "1" ]] && [[ "${HAS_SUDO}" == "1" ]]; then
    log_step "===== 1.5. 安装 VNC ====="
    install_vnc
else
    log_step "===== 1.5. 跳过 VNC 安装 ====="
fi

# ============================================================================
# 2. 安装 Miniconda
# ============================================================================
if [[ "${SKIP_CONDA}" != "1" ]]; then
    log_step "===== 2. 安装 Miniconda ====="
    
    if [[ -x "${CONDA_DIR}/bin/conda" ]]; then
        log_info "Miniconda 已安装在 ${CONDA_DIR}，跳过安装"
    else
        MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        
        # 根据是否使用国内镜像选择下载源
        if [[ "${USE_CN_MIRROR}" == "1" ]]; then
            download_with_fallback /tmp/miniconda.sh \
                "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/${MINICONDA_INSTALLER}" \
                "https://mirrors.ustc.edu.cn/anaconda/miniconda/${MINICONDA_INSTALLER}" \
                "https://mirrors.bfsu.edu.cn/anaconda/miniconda/${MINICONDA_INSTALLER}" \
                "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"
        else
            download_with_fallback /tmp/miniconda.sh \
                "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"
        fi
        
        chmod +x /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
        rm -f /tmp/miniconda.sh
        log_info "Miniconda 安装完成: ${CONDA_DIR}"
    fi

    # 配置 conda mirror
    if [[ "${USE_CN_MIRROR}" == "1" ]]; then
        log_info "配置 conda 国内镜像源（清华）"
        "${CONDA_DIR}/bin/conda" config --set show_channel_urls yes
        "${CONDA_DIR}/bin/conda" config --remove-key default_channels 2>/dev/null || true
        "${CONDA_DIR}/bin/conda" config --append channels "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
        "${CONDA_DIR}/bin/conda" config --append channels "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r"
        "${CONDA_DIR}/bin/conda" config --append channels "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2"
        "${CONDA_DIR}/bin/conda" config --set custom_channels.conda-forge \
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud"
        # nvidia channel 使用官方源（清华镜像的 nvidia channel 经常不可用）
        # "${CONDA_DIR}/bin/conda" config --set custom_channels.nvidia \
        #     "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud"
    fi

    # 同意 conda tos
    "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel \
        https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel \
        https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    "${CONDA_DIR}/bin/conda" clean -afy

    # 创建 conda 环境
    if "${CONDA_DIR}/bin/conda" env list | grep -q "^${CONDA_ENV} "; then
        log_info "conda 环境 '${CONDA_ENV}' 已存在，跳过创建"
    else
        log_info "创建 conda 环境 '${CONDA_ENV}' (python=${PYTHON_VERSION})"
        "${CONDA_DIR}/bin/conda" create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}"
        log_info "conda 环境创建完成"
    fi

    # 安装 PyTorch（通过 pip 安装 CUDA 版本以确保正确的 CUDA 支持）
    log_info "安装 PyTorch ${TORCH_VERSION}+cu${CUDA_VERSION}"
    "${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip" install \
        --retries 20 --default-timeout 120 \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION/./}"
    "${CONDA_DIR}/bin/conda" clean -afy
    
    log_info "PyTorch 安装完成"

    # 验证 PyTorch 安装
    log_info "验证 PyTorch 安装..."
    "${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" -c \
        "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
else
    log_step "===== 2. 跳过 Miniconda 安装 ====="
fi

# ============================================================================
# 3. 安装 pip 依赖
# ============================================================================
if [[ "${SKIP_PIP}" != "1" ]]; then
    log_step "===== 3. 安装 pip 依赖 ====="

    PIP_BIN="${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip"

    # 升级 pip
    "${PIP_BIN}" install --upgrade pip

    # 配置 pip 镜像
    if [[ "${USE_CN_MIRROR}" == "1" ]]; then
        log_info "配置 pip 国内镜像源（清华）"
        "${PIP_BIN}" config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        "${PIP_BIN}" config set global.trusted-host pypi.tuna.tsinghua.edu.cn
    fi

    REQUIRMENTS_FILE="${PROJECT_DIR}/requirements.txt"
    if [[ ! -f "${REQUIRMENTS_FILE}" ]]; then
        log_error "找不到 requirements.txt: ${REQUIRMENTS_FILE}"
        exit 1
    fi

    # 移除 torch/torchvision 行和 -f 行（因为已通过 conda 安装），安装其余依赖
    log_info "解析 requirements.txt 并安装轻量依赖（跳过 torch/torchvision，已通过 conda 安装）"
    
    # 创建临时文件，去掉 torch、torchvision 和 -f 行
    TMP_REQUIREMENTS="/tmp/requirements_light.txt"
    awk 'BEGIN{skip=0} /^-f[[:space:]]/{next} /^torch([<=>]|$)/{next} /^torchvision([<=>]|$)/{next} {print}' \
        "${REQUIRMENTS_FILE}" > "${TMP_REQUIREMENTS}"
    
    log_info "开始安装 pip 依赖..."
    while IFS= read -r requirement; do
        case "$requirement" in
            ""|\#*) continue ;;
            *)
                pip_install_with_fallback "${requirement}" || {
                    log_error "pip 安装失败: ${requirement}"
                    exit 1
                }
                ;;
        esac
    done < "${TMP_REQUIREMENTS}"
    
    rm -f "${TMP_REQUIREMENTS}"
    
    # 生成 freeze 文件
    log_info "生成 requirements.freeze.txt"
    "${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip" freeze > "${PROJECT_DIR}/requirements.freeze.txt"
    
    log_info "pip 依赖安装完成"
else
    log_step "===== 3. 跳过 pip 依赖安装 ====="
fi

# ============================================================================
# 4. 安装 Isaac Gym 和 IsaacGymEnvs
# ============================================================================
if [[ "${SKIP_ISAAC}" != "1" ]]; then
    log_step "===== 4. 安装 Isaac Gym 和 IsaacGymEnvs ====="

    PIP_BIN="${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip"
    WORKSPACE_DIR="${PROJECT_DIR}"

    # 4.1 自动发现 Isaac Gym 源码
    ISAACGYM_SRC_DIR=""
    
    # 优先搜索 isaacgym1 目录
    if [[ -f "${WORKSPACE_DIR}/isaacgym1/isaacgym/python/setup.py" ]]; then
        ISAACGYM_SRC_DIR="${WORKSPACE_DIR}/isaacgym1/isaacgym/python"
        log_info "找到 Isaac Gym 源码: ${ISAACGYM_SRC_DIR}"
    fi

    # 如果没找到，尝试解压压缩包
    if [[ -z "${ISAACGYM_SRC_DIR}" ]]; then
        for archive in "${WORKSPACE_DIR}/issacgym.tar.xz" "${WORKSPACE_DIR}/isaacgym.tar.xz"; do
            if [[ -f "${archive}" ]]; then
                log_info "找到 Isaac Gym 压缩包: ${archive}，正在解压..."
                rm -rf /tmp/isaacgym_extract "${WORKSPACE_DIR}/isaacgym1"
                mkdir -p /tmp/isaacgym_extract "${WORKSPACE_DIR}/isaacgym1"
                tar -xJf "${archive}" -C /tmp/isaacgym_extract
                
                ISAACGYM_SETUP_PATH="$(find /tmp/isaacgym_extract -maxdepth 8 -path '*/isaacgym/python/setup.py' | head -n 1 || true)"
                if [[ -z "${ISAACGYM_SETUP_PATH}" ]]; then
                    log_error "无法在压缩包中找到 isaacgym/python/setup.py"
                    rm -rf /tmp/isaacgym_extract
                    continue
                fi
                
                ISAACGYM_ROOT_DIR="$(dirname "$(dirname "$(dirname "${ISAACGYM_SETUP_PATH}")")")"
                cp -a "${ISAACGYM_ROOT_DIR}/." "${WORKSPACE_DIR}/isaacgym1/"
                rm -rf /tmp/isaacgym_extract
                ISAACGYM_SRC_DIR="${WORKSPACE_DIR}/isaacgym1/isaacgym/python"
                log_info "解压完成: ${ISAACGYM_SRC_DIR}"
                break
            fi
        done
    fi

    # 如果还没找到，询问用户或报错
    if [[ -z "${ISAACGYM_SRC_DIR}" ]]; then
        log_error "找不到 Isaac Gym 源码。"
        log_error "请将 Isaac Gym 源码放置在以下任一位置："
        log_error "  1. ${WORKSPACE_DIR}/isaacgym1/isaacgym/python/setup.py"
        log_error "  2. ${WORKSPACE_DIR}/issacgym.tar.xz（或 isaacgym.tar.xz）压缩包"
        log_error "你也可以设置环境变量 SKIP_ISAAC=1 跳过此步骤"
        exit 1
    fi

    # 安装 Isaac Gym
    log_info "安装 Isaac Gym（可编辑模式，--no-deps）..."
    "${PIP_BIN}" install --no-deps -e "${ISAACGYM_SRC_DIR}"
    log_info "Isaac Gym 安装完成"

    # 4.2 安装 IsaacGymEnvs
    ISAACGYMENVS_DIR="${WORKSPACE_DIR}/IsaacGymEnvs"
    
    if [[ ! -f "${ISAACGYMENVS_DIR}/setup.py" ]] && [[ ! -f "${ISAACGYMENVS_DIR}/pyproject.toml" ]]; then
        log_info "IsaacGymEnvs 源码缺失，正在克隆: ${ISAACGYMENVS_GIT_URL}"
        rm -rf "${ISAACGYMENVS_DIR}"
        git clone --depth 1 "${ISAACGYMENVS_GIT_URL}" "${ISAACGYMENVS_DIR}"
    fi

    if [[ -f "${ISAACGYMENVS_DIR}/setup.py" ]] || [[ -f "${ISAACGYMENVS_DIR}/pyproject.toml" ]]; then
        log_info "安装 IsaacGymEnvs（可编辑模式，--no-deps）..."
        "${PIP_BIN}" install --no-deps -e "${ISAACGYMENVS_DIR}"
        log_info "IsaacGymEnvs 安装完成"
    else
        log_error "IsaacGymEnvs 项目元数据未找到: ${ISAACGYMENVS_DIR}"
        exit 1
    fi

    # 验证安装
    log_info "验证 Isaac Gym 安装..."
    "${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" -c \
        "import isaacgym; print('Isaac Gym 导入成功')" || {
        log_warn "Isaac Gym 导入验证失败，请检查 LD_LIBRARY_PATH 和依赖"
    }
    
else
    log_step "===== 4. 跳过 Isaac Gym/Envs 安装 ====="
fi

# ============================================================================
# 4.5. 安装 legged_gym 和 rsl_rl
# ============================================================================
if [[ "${SKIP_LEGGED}" != "1" ]]; then
    log_step "===== 4.5. 安装 legged_gym 和 rsl_rl ====="
    
    PIP_BIN="${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip"
    
    RSL_RL_DIR="${PROJECT_DIR}/rsl_rl"
    if [[ -d "${RSL_RL_DIR}" ]]; then
        log_info "安装 rsl_rl（可编辑模式）..."
        "${PIP_BIN}" install --no-deps -e "${RSL_RL_DIR}"
        log_info "rsl_rl 安装完成"
    else
        log_warn "rsl_rl 目录不存在: ${RSL_RL_DIR}"
    fi
    
    LEGGED_GYM_DIR="${PROJECT_DIR}/legged_gym"
    if [[ -d "${LEGGED_GYM_DIR}" ]]; then
        log_info "安装 legged_gym（可编辑模式）..."
        "${PIP_BIN}" install --no-deps -e "${LEGGED_GYM_DIR}"
        log_info "legged_gym 安装完成"
    else
        log_warn "legged_gym 目录不存在: ${LEGGED_GYM_DIR}"
    fi
    
    log_info "验证 legged_gym 和 rsl_rl 安装..."
    "${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" -c \
        "import legged_gym; print('legged_gym 导入成功')" || \
        log_warn "legged_gym 导入验证失败"
    "${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" -c \
        "import rsl_rl; print('rsl_rl 导入成功')" || \
        log_warn "rsl_rl 导入验证失败"
else
    log_step "===== 4.5. 跳过 legged_gym/rsl_rl 安装 ====="
fi

# ============================================================================
# 5. 安装 MuJoCo 2.1.0
# ============================================================================
if [[ "${SKIP_MUJOCO}" != "1" ]]; then
    log_step "===== 5. 安装 MuJoCo 2.1.0 ====="

    if [[ -d "${MUJOCO_DIR}/mujoco210" ]]; then
        log_info "MuJoCo 2.1.0 已安装在 ${MUJOCO_DIR}/mujoco210，跳过安装"
    else
        MUJOCO_TAR="mujoco210-linux-x86_64.tar.gz"
        MUJOCO_URL="https://mujoco.org/download/${MUJOCO_TAR}"
        
        mkdir -p "${MUJOCO_DIR}"
        
        log_info "下载 MuJoCo 2.1.0..."
        wget --tries=5 --waitretry=2 --timeout=30 \
            -c -O "${MUJOCO_DIR}/${MUJOCO_TAR}" "${MUJOCO_URL}" || {
            log_warn "MuJoCo 官方下载失败，尝试备用地址..."
            download_with_fallback "${MUJOCO_DIR}/${MUJOCO_TAR}" \
                "https://github.com/deepmind/mujoco/releases/download/2.1.0/${MUJOCO_TAR}" \
                "https://ghproxy.com/https://github.com/deepmind/mujoco/releases/download/2.1.0/${MUJOCO_TAR}"
        }
        
        log_info "解压 MuJoCo..."
        tar -zxvf "${MUJOCO_DIR}/${MUJOCO_TAR}" -C "${MUJOCO_DIR}"
        log_info "MuJoCo 解压完成: ${MUJOCO_DIR}/mujoco210"
    fi

    # 验证
    if [[ -f "${MUJOCO_DIR}/mujoco210/bin/libmujoco210.so" ]]; then
        log_info "MuJoCo 验证通过: libmujoco210.so 存在"
    else
        log_warn "未找到 libmujoco210.so，请检查 MuJoCo 安装"
    fi
else
    log_step "===== 5. 跳过 MuJoCo 安装 ====="
fi

# ============================================================================
# 6. 配置环境变量
# ============================================================================
log_step "===== 6. 配置环境变量 ====="

CONDA_LIB="${CONDA_DIR}/envs/${CONDA_ENV}/lib"
SYSTEM_LIB="/usr/lib/x86_64-linux-gnu"
MUJOCO_BIN="${MUJOCO_DIR}/mujoco210/bin"

mkdir -p "${PROJECT_DIR}/sh"

# 写入环境变量配置脚本
ENV_SCRIPT="${PROJECT_DIR}/sh/env.sh"
cat > "${ENV_SCRIPT}" << ENVEOF
#!/usr/bin/env bash
# 自动生成的环境变量配置，由 deploy.sh 生成
# 使用方式：source \$(dirname "\${BASH_SOURCE[0]}")/env.sh

export MUJOCO_GL=egl
export MUJOCO_DIR="${MUJOCO_DIR}"

# conda 路径
export PATH="${CONDA_DIR}/bin:\${PATH}"

# 动态库搜索路径
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_DIR}/lib:${SYSTEM_LIB}:${MUJOCO_BIN}:\${LD_LIBRARY_PATH:-}"

# 显示设置（如需 VNC）
export DISPLAY="\${DISPLAY:-:1}"

# VirtualGL 和 TurboVNC 路径
export VGL_HOME="/opt/VirtualGL"
export TURBOVNC_HOME="/opt/TurboVNC"
export PATH="\${TURBOVNC_HOME}/bin:\${VGL_HOME}/bin:\${PATH}"
export VGL_GLFLUSH=1
export VGL_VSYNC=1

# 激活 conda 环境
if [[ "\$1" != "--no-activate" ]]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh" && conda activate ${CONDA_ENV}
fi

echo "[env] conda env: ${CONDA_ENV}"
echo "[env] MUJOCO_GL: \${MUJOCO_GL}"
echo "[env] MUJOCO_DIR: \${MUJOCO_DIR}"
echo "[env] LD_LIBRARY_PATH configured"
ENVEOF

chmod +x "${ENV_SCRIPT}"
log_info "环境变量脚本已生成: ${ENV_SCRIPT}"

# 写入 ~/.bashrc 追加（可选）
BASHRC_APPEND="${PROJECT_DIR}/sh/env.sh"
log_info "如需每次登录自动加载环境，请将以下行添加到 ~/.bashrc："
log_info "  source ${BASHRC_APPEND} --no-activate && conda activate ${CONDA_ENV}"

# ============================================================================
# 7. 创建目录和链接
# ============================================================================
log_step "===== 7. 创建目录和符号链接 ====="

# 创建 checkpoints 和 logs 目录
mkdir -p "${PROJECT_DIR}/checkpoints" "${PROJECT_DIR}/logs"
log_info "已创建 checkpoints/ 和 logs/ 目录"

# 为 libpython3.8 创建符号链接（Isaac Gym gym_38.so 依赖）
PYTHON38_SO="${CONDA_LIB}/libpython3.8.so.1.0"
if [[ -f "${PYTHON38_SO}" ]] && [[ "${HAS_SUDO}" == "1" ]]; then
    if [[ ! -e "${SYSTEM_LIB}/libpython3.8.so.1.0" ]]; then
        log_info "创建 libpython3.8 符号链接..."
        sudo mkdir -p "${SYSTEM_LIB}" 2>/dev/null || true
        sudo ln -sf "${PYTHON38_SO}" "${SYSTEM_LIB}/libpython3.8.so.1.0" 2>/dev/null || true
        if [[ -e "${CONDA_LIB}/libpython3.8.so" ]]; then
            sudo ln -sf "${CONDA_LIB}/libpython3.8.so" "${SYSTEM_LIB}/libpython3.8.so" 2>/dev/null || true
        fi
        sudo ldconfig 2>/dev/null || true
        log_info "libpython3.8 符号链接创建完成"
    fi
fi

# 创建 VNC 启动脚本
VNC_START_SCRIPT="${PROJECT_DIR}/sh/start_vnc.sh"
cat > "${VNC_START_SCRIPT}" << 'VNCEOF'
#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:1
export VGL_GLFLUSH=1
export VGL_VSYNC=1
export MUJOCO_GL=egl

TURBOVNC_BIN="/opt/TurboVNC/bin"
VNC_PASSWD_FILE="${HOME}/.vnc/passwd"

# 检查 VNC 密码是否已设置
if [[ ! -f "${VNC_PASSWD_FILE}" ]]; then
    echo "首次使用 VNC，请设置密码（至少6位）:"
    ${TURBOVNC_BIN}/vncpasswd
fi

# 停止已有的 VNC 服务
if ${TURBOVNC_BIN}/vncserver -list 2>/dev/null | grep -q ":1"; then
    echo "停止已有的 VNC :1..."
    ${TURBOVNC_BIN}/vncserver -kill :1 2>/dev/null || true
    sleep 1
fi

echo "启动 TurboVNC server..."
${TURBOVNC_BIN}/vncserver :1 -geometry 1920x1080 -depth 24 -wm xfce

echo ""
echo "VNC 服务已启动"
echo "连接地址: <your_server_ip>:5901"
echo "如需停止 VNC: ${TURBOVNC_BIN}/vncserver -kill :1"
VNCEOF

chmod +x "${VNC_START_SCRIPT}"
log_info "VNC 启动脚本已生成: ${VNC_START_SCRIPT}"

# ============================================================================
# 8. 验证
# ============================================================================
log_step "===== 8. 验证安装 ====="

echo ""
echo "========================================"
echo "  验证结果"
echo "========================================"
echo ""

# 验证 Python
echo "--- Python ---"
"${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" --version || log_warn "Python 验证失败"

# 验证 PyTorch
echo "--- PyTorch ---"
"${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" -c \
    "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')" 2>/dev/null || \
    log_warn "PyTorch 验证失败"

# 验证 GPU
echo "--- GPU ---"
if [[ "${NO_GPU_VERIFY}" == "1" ]]; then
    echo "GPU 验证已跳过 (NO_GPU_VERIFY=1)"
elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
else
    log_warn "nvidia-smi 不可用（无卡模式）"
fi

# 验证 Isaac Gym
echo "--- Isaac Gym ---"
if [[ "${NO_GPU_VERIFY}" == "1" ]]; then
    echo "Isaac Gym 验证已跳过 (NO_GPU_VERIFY=1)"
else
    "${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" -c \
        "import isaacgym; print('Isaac Gym 导入成功')" 2>/dev/null || \
        log_warn "Isaac Gym 验证失败（请先运行: source ${PROJECT_DIR}/sh/env.sh）"
fi

# 验证 MuJoCo
echo "--- MuJoCo ---"
if [[ -f "${MUJOCO_DIR}/mujoco210/bin/libmujoco210.so" ]]; then
    echo "MuJoCo 2.1.0 已安装: ${MUJOCO_DIR}/mujoco210"
else
    log_warn "MuJoCo 未找到"
fi

# 验证 IsaacGymEnvs
echo "--- IsaacGymEnvs ---"
if [[ -d "${PROJECT_DIR}/IsaacGymEnvs" ]]; then
    echo "IsaacGymEnvs 已就绪: ${PROJECT_DIR}/IsaacGymEnvs"
else
    log_warn "IsaacGymEnvs 未找到"
fi

# 验证 legged_gym
echo "--- legged_gym ---"
"${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" -c \
    "import legged_gym; print('legged_gym 导入成功')" 2>/dev/null || \
    log_warn "legged_gym 验证失败（请先运行: source ${PROJECT_DIR}/sh/env.sh）"

# 验证 rsl_rl
echo "--- rsl_rl ---"
"${CONDA_DIR}/envs/${CONDA_ENV}/bin/python" -c \
    "import rsl_rl; print('rsl_rl 导入成功')" 2>/dev/null || \
    log_warn "rsl_rl 验证失败（请先运行: source ${PROJECT_DIR}/sh/env.sh）"

# 验证 VNC
echo "--- VNC ---"
if command -v vncserver >/dev/null 2>&1 || [[ -x /opt/TurboVNC/bin/vncserver ]]; then
    echo "TurboVNC 已安装"
else
    log_warn "TurboVNC 未安装"
fi

echo ""
echo "========================================"
echo "  部署完成"
echo "========================================"
echo ""
echo "快速开始："
echo ""
echo "  1. 加载环境变量:"
echo "     source ${PROJECT_DIR}/sh/env.sh"
echo ""
echo "  2. 运行烟雾测试（Cartpole 训练）:"
echo "     cd ${PROJECT_DIR}/IsaacGymEnvs"
echo "     conda run -n ${CONDA_ENV} python isaacgymenvs/train.py task=Cartpole sim_device=cuda:0 rl_device=cuda:0 headless=True max_iterations=50"
echo ""
echo "  3. 启动 VNC 远程桌面（如已安装）:"
echo "     bash ${PROJECT_DIR}/sh/start_vnc.sh"
echo ""
echo "  4. 连接到 VNC 服务器:"
echo "     使用 VNC Viewer 连接到 <your_server_ip>:5901"
echo "     注意：需在云服务器安全组中开放 TCP 5901 端口"
echo ""
echo "  5. 在 VNC 中打开终端，运行训练测试:"
echo "     source ${PROJECT_DIR}/sh/env.sh"
echo "     cd ${PROJECT_DIR}/legged_gym/scripts"
echo "     python train.py"
echo ""
echo "  6. 注意事项:"
echo "     - MuJoCo 2.1+ 通常无需 license key"
echo "     - 如遇到 libpython3.8 加载错误，确保 LD_LIBRARY_PATH 包含 conda 的 lib 目录"
echo "     - 大模型/数据集请勿放在项目目录中，建议单独存放并挂载"
echo "     - 无卡部署：使用 NO_GPU_VERIFY=1 可跳过GPU验证但仍安装CUDA依赖"
echo "     - 迁移到有卡机器后，运行 'nvidia-smi' 确认GPU可用后再训练"
echo ""