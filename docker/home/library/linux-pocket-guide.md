# Linux pocket guide

## Files
- `pwd` where am I; `ls -la DIR` list; `cd DIR` move; `tree DIR` show a tree.
- `cat FILE` print; `head -n 20 FILE`, `tail -n 20 FILE` the start or end; `wc -l FILE` count lines.
- `mkdir -p DIR`; `cp A B`; `mv A B`; `rm FILE`; `rm -r DIR` (no undo).
- `echo "text" > FILE` write; `echo "text" >> FILE` append.
- `find DIR -name "*.txt"` find files; `grep -rin WORD DIR` find text in files.

## Text
- `sort`, `uniq -c`, `cut -d, -f2`, `tr a-z A-Z`, `sed 's/old/new/g'`, `awk '{print $1}'`.
- `cmd1 | cmd2` sends one command's output into the next.

## Numbers
- `python3 -c "print(17*23)"`; `echo "scale=3; 10/3" | bc`.

## Programs
- Write a script: `printf '%s\n' 'line 1' 'line 2' > ~/work/x.sh`, then `bash ~/work/x.sh`.
- Python: `python3 FILE.py`; `python3 -c "CODE"`.
- `ps aux` running programs; `kill PID` stop one; `sleep 5` wait.

## The system
- `whoami`, `hostname`, `date`, `df -h`, `free -h`, `uname -a`.
- `sudo COMMAND` run as root. `sudo apt-get install -y PACKAGE` install; `apt-cache search WORD` find.

## The network
- `ask-web QUESTION` look something up; `curl -s URL`; `w3m -dump URL` read a page as text;
  `ping -c 3 HOST`.
