#!/usr/bin/env bash
set -euo pipefail

: "${1:?Usage: install-system-deps.sh package...}"
apt_options=(-o Acquire::Retries=2 -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15)

install_packages() {
  sudo timeout --kill-after=10s 90s apt-get "${apt_options[@]}" \
    -o APT::Update::Error-Mode=any update &&
    sudo timeout --kill-after=10s 90s apt-get "${apt_options[@]}" install -y "$@"
}

if install_packages "$@"; then
  exit 0
fi

echo "::warning::APT setup failed; retrying with alternate Ubuntu mirrors."
printf '%s\tpriority:%s\n' \
  'https://mirrors.edge.kernel.org/ubuntu/' 1 \
  'https://archive.ubuntu.com/ubuntu/' 2 \
  'https://security.ubuntu.com/ubuntu/' 3 |
  sudo tee /etc/apt/apt-mirrors.txt >/dev/null
install_packages "$@"
