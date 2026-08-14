# Relaying visual information from the Supernote Manta

The user's Supernote Manta e-reader/tablet can screen-mirror over the LAN. Its viewer app serves
a plain HTTP page at `http://<tablet-ip>:8080/` with an embedded MJPEG stream — a convenient way
for the user to hand-sketch a figure and have Claude look at it, without any file transfer.

## Why `WebFetch` doesn't work here

`WebFetch` runs off-machine (remote infra), not on the user's LAN, so a private/RFC1918 address
like `10.x.x.x` does not resolve to the tablet — it either times out or hits an unrelated host
(observed once as a bare `WRONG_VERSION_NUMBER` TLS error, i.e. it reached *something*, just not
the tablet). **Use the `Bash` tool instead** — it executes locally on the user's machine, on the
same LAN as the tablet, so plain `curl` reaches it directly.

## Recipe: capture one still frame

1. **Confirm reachability and find the stream endpoint.**
   ```
   curl -sS -m 5 -D - -o page.html "http://<tablet-ip>:8080/"
   ```
   The root page is an HTML viewer; look for `<img id="stream" src="screencast.mjpeg"/>` — the
   actual feed is at `http://<tablet-ip>:8080/screencast.mjpeg`.

2. **Read the multipart boundary from the headers.** The stream is
   `multipart/x-mixed-replace; boundary=<token>`, and the token is generated per session — don't
   hardcode a previously seen value, re-fetch it each time:
   ```
   curl -sS -m 2 -D - -o /dev/null "http://<tablet-ip>:8080/screencast.mjpeg"
   ```
   look for `Content-Type: multipart/x-mixed-replace; boundary=XXXXXXXXXXXXXXXXXXXX`.

3. **Capture a few seconds of the raw stream into a file** (long enough to contain at least one
   complete part — the first part in the capture is often mid-frame/truncated, so plan to skip
   it):
   ```
   curl -sS -m 3 "http://<tablet-ip>:8080/screencast.mjpeg" -o chunk.bin
   ```

4. **Extract one complete frame properly — do not just scan for JPEG SOI/EOI markers
   (`0xFFD8`/`0xFFD9`) in the raw bytes.** `0xFF` bytes occur constantly inside compressed image
   data and multipart headers, so a naive marker search finds false positives and yields a
   corrupt image (this failed on the first attempt). Instead, split on the real boundary and use
   each part's own `Content-Length` header:
   ```python
   import re
   data = open("chunk.bin", "rb").read()
   boundary = b"--" + b"<token from step 2>"
   for part in data.split(boundary):
       hdr_end = part.find(b"\r\n\r\n")
       if hdr_end == -1:
           continue
       header, body = part[:hdr_end], part[hdr_end + 4:]
       m = re.search(rb"Content-Length:\s*(\d+)", header, re.IGNORECASE)
       if not m:
           continue
       length = int(m.group(1))
       if len(body) >= length:
           open("frame.jpg", "wb").write(body[:length])
           break  # first complete part is enough
   ```

5. **`Read` the extracted file.** Note: the part header claims `Content-Type: image/jpeg`, but
   the bytes observed in practice were actually PNG-encoded. Doesn't matter — the `Read` tool
   sniffs actual content and ignores the file extension, so no special-casing is needed; just
   point it at the file.

## Caveats

- This is an **on-demand single-frame snapshot**, not continuous monitoring — re-run the recipe
  each time an updated view is needed. There is no standing "watch this stream" mechanism.
- Save capture artifacts under the session scratchpad, not inside the repo.
