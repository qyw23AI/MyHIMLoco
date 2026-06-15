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