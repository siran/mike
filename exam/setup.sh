#!/usr/bin/env bash
# Prepare a fresh home for the exam: no map, and the few files the questions need (exam/questions.txt).
#   exam/setup.sh VOLUME
set -euo pipefail
vol=${1:?usage: exam/setup.sh VOLUME}
docker volume rm -f "$vol" >/dev/null 2>&1 || true
docker run --rm -u mike -w /home/mike -v "$vol:/home/mike" "${MIKE_IMAGE:-mike}" bash -c '
set -e
rm -f ~/WHERE-TO-LOOK.md
mkdir -p ~/notes ~/work/projects/old-letters
printf "%s\n" "Dear future me," "" "My favourite book is The Little Prince." "" "- 2019" > ~/work/projects/old-letters/2019-letter.txt
printf "%s\n" "# written 2024-03-01" "The office phone is 555-0101." > ~/notes/phone-2024.md
printf "%s\n" "# written 2026-09-10" "The office phone changed to 555-0199." > ~/notes/phone-2026.md
printf "%s\n" "Quarterly report" "Sales: 120" "Costs: 80" "Total: 40" > ~/work/report-v1.txt
printf "%s\n" "Quarterly report" "Sales: 120" "Costs: 80" "Total: 45" > ~/work/report-v2.txt
echo "home ready: $(find ~ -type f -not -path "*/.*" | sort | tr "\n" " ")"
'
