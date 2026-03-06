# AIDZero Android App

Native Android client for talking to an AIDZero core running on the same local network.

## Current scope

- Configure the core IP or full base URL from the app.
- Verify `/health`.
- Send prompts to `/engine/run_event`.
- Reset the session through `/engine/session/reset`.
- Store the selected core URL locally on the phone.
- Cleartext HTTP is enabled because this is LAN-only for now.

## Build debug APK

From this folder:

```bash
./gradlew assembleDebug
```

Expected output:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## Notes

- The core must be reachable from the phone on the same LAN.
- If you run the core on a different host, expose it on a LAN IP, not only `127.0.0.1`.
- The app currently targets the existing core HTTP API directly.
- You need a local Android SDK configured through `ANDROID_HOME`, `ANDROID_SDK_ROOT`, or `local.properties`.
