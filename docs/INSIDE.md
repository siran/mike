# Inside MIK-e: talk to him, watch him, look around his computer

Everything runs from a WSL (Ubuntu) terminal in the repo: `cd ~/src/mike`. MIK-e lives in a Docker container
built from the `mike` image (`scripts/build.sh`). His home, `/home/mike`, is a named Docker volume, so his
notes and conversations survive between runs. The container never sees your WSL home, keys or the Docker socket.

## Talk to him

```bash
MIKE_HOME_VOLUME=mike-home-an scripts/chat.sh                                              # 360M (default)
MIKE_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct MIKE_HOME_VOLUME=mike-home-an scripts/chat.sh   # 1.7B
```

You type at `you>`. You see what he says, every command he runs (`$ …`) and its output. `/new` starts a fresh
conversation (his notes stay), `/quit` leaves. `MIKE_HOME_VOLUME` picks whose home he wakes up in: use your
own (e.g. `mike-home-an`) so your chats don't mix with the test runs.

## Go inside his computer

A shell as `mike`, in his home (he has passwordless `sudo`):

```bash
docker run --rm -it -u mike -w /home/mike -v mike-home-an:/home/mike mike bash
```

Inside, the useful places:

```bash
ls ~/notes                      # what he wrote down (memory he chose to keep)
ls -t ~/conversations | head    # every session: .txt to read, .jsonl with raw model output and timings
less ~/conversations/<file>.txt
cat ~/WHERE-TO-LOOK.md          # his map of where information lives
cat ~/VALUES.md                 # his values (draft)
ls ~/library ~/work             # books he can read; his scratch space
```

Or jump into a chat that's already running:

```bash
docker exec -it -u mike $(docker ps --filter ancestor=mike --format '{{.ID}} {{.Command}}' | grep 'mike chat' | cut -d' ' -f1) bash
```

## Watch the tests

```bash
docker ps --filter ancestor=mike                                   # what's running
docker logs -f $(docker ps -q --filter ancestor=mike | head -1)    # stream one live
```

Scripted conversations live in `tests/conversations/`. Run one against a throwaway home (the script goes on
stdin, because the container doesn't see the repo):

```bash
MIKE_HOME_VOLUME=mike-home-test scripts/chat.sh converse --script - < tests/conversations/remember-add.txt
```

## Read a conversation without starting him

```bash
docker run --rm -v mike-home-an:/home/mike mike sh -c 'cat "/home/mike/conversations/$(ls -t /home/mike/conversations | grep txt | head -1)"'
```

## Start over

His home is just a volume: `docker volume rm mike-home-an` gives him a blank one next time (his notes and
conversations are gone; the model cache, `mike-hf`, is separate and stays).
