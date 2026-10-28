# MIK-e's computer: a Debian with the tools a person at a Linux terminal has, man pages included.
FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8

# Slim images strip man pages and docs: drop that rule, then reinstall what is already here to restore them.
# Apt lists are kept so `sudo apt-get install` works without an update first.
RUN rm /etc/dpkg/dpkg.cfg.d/docker \
 && apt-get update \
 && apt-get -y upgrade \
 && apt-get install -y --no-install-recommends \
      bash coreutils findutils grep sed gawk less file tree procps psmisc \
      curl wget ca-certificates git w3m lynx jq bc \
      python3 python3-venv python3-pip \
      man-db manpages groff-base sudo nano vim-tiny unzip xz-utils iputils-ping \
 && apt-get install -y --reinstall $(dpkg-query -W -f='${binary:Package}\n') \
 && mandb --quiet --create

# The kernel's own Python (torch, transformers) lives apart from the system python3 mike uses.
RUN python3 -m venv /opt/mike/venv \
 && /opt/mike/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && /opt/mike/venv/bin/pip install --no-cache-dir transformers==4.57.6 "safetensors>=0.4"

# His commands have no keyboard or terminal: nothing can answer apt's Y/n prompt or watch a progress bar.
RUN useradd --create-home --shell /bin/bash mike \
 && printf 'APT::Get::Assume-Yes "true";\nDpkg::Use-Pty "0";\n' > /etc/apt/apt.conf.d/90mike-no-keyboard \
 && printf 'mike ALL=(ALL:ALL) NOPASSWD:ALL\nDefaults:mike !use_pty\nDefaults:mike env_keep += "DEBIAN_FRONTEND"\n' \
      > /etc/sudoers.d/mike \
 && chmod 0440 /etc/sudoers.d/mike

COPY docker/ask-web /usr/local/bin/ask-web
COPY docker/mike /usr/local/bin/mike
COPY --chown=mike:mike docker/home/ /home/mike/
RUN chmod 755 /usr/local/bin/ask-web /usr/local/bin/mike \
 && install -d -o mike -g mike /home/mike/notes /home/mike/library /home/mike/work \
 && install -d -o root -g root -m 755 /home/mike/conversations \
 && install -d -m 755 /var/cache/huggingface

# The clean kernel: every container starts from this copy.
COPY src /opt/mike/src
COPY tests /opt/mike/tests
RUN chown -R root:root /opt/mike/src /opt/mike/tests && chmod -R a+rX,go-w /opt/mike/src /opt/mike/tests

ENV HF_HOME=/var/cache/huggingface
WORKDIR /opt/mike
CMD ["mike", "chat"]
