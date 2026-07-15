"""
Refresh auth.json using credential.txt
Tries password login first; falls back to manual JWT from browser if Biometrics required.
"""
import json, os, sys, requests, base64, time
from pathlib import Path
from requests.auth import HTTPBasicAuth

SCRIPT_DIR = Path(__file__).parent
GEN_TWO = SCRIPT_DIR / "generation_two"
CRED_FILE = GEN_TWO / "credential.txt"
AUTH_FILE = GEN_TWO / "auth.json"
ROOT_AUTH = SCRIPT_DIR / "auth.json"
API = "https://api.worldquantbrain.com"

def load_credentials():
    """Read email/password from credential.txt"""
    if not CRED_FILE.exists():
        print(f"[ERR] credential.txt not found at {CRED_FILE}")
        return None, None
    with open(CRED_FILE) as f:
        lines = f.read().strip().split('\n')
    if len(lines) < 2:
        print(f"[ERR] credential.txt format: email on line 1, password on line 2")
        return None, None
    return lines[0].strip(), lines[1].strip()


def save_jwt_auth_json(jwt_token, path):
    """Save JWT to auth.json in the format fetch_all_datafields.py expects"""
    import calendar
    exp = int(time.time()) + 86400  # 24h from now
    auth_data = {
        "cookies": [{
            "name": "t",
            "value": jwt_token,
            "domain": ".worldquantbrain.com",
            "path": "/",
            "expires": exp,
            "httpOnly": False,
            "secure": True,
            "sameSite": "Lax"
        }],
        "origins": []
    }
    with open(path, 'w') as f:
        json.dump(auth_data, f, indent=2)
    print(f"[OK] auth.json saved to {path}")


def try_password_login(email, password):
    """Try HTTP Basic Auth login and extract JWT"""
    print(f"\n[1] Trying password login for {email}...")
    sess = requests.Session()
    auth = HTTPBasicAuth(email, password)

    r = sess.post(f"{API}/authentication", auth=auth, timeout=15)
    print(f"    HTTP {r.status_code}")

    if r.status_code == 201:
        print("[OK] Password login successful!")

        # Try to get JWT by calling /users/self
        r2 = sess.get(f"{API}/users/self", timeout=15)
        if r2.status_code == 200:
            user = r2.json()
            print(f"    User: {user.get('email')} | Level: {user.get('geniusLevel')}")

        # Extract JWT from cookies if present
        for cookie in sess.cookies:
            if cookie.name == 't' and cookie.value.startswith('eyJ'):
                print(f"[OK] Got JWT from session cookies")
                return cookie.value

        # If no JWT cookie, try getting one via /token endpoint
        print("    No JWT cookie found, trying /token endpoint...")
        r3 = sess.get(f"{API}/token", auth=auth, timeout=15)
        if r3.status_code == 201:
            data = r3.json()
            jwt = data.get('token') or data.get('jwt') or data.get('access_token')
            if jwt:
                print("[OK] Got JWT from /token endpoint")
                return jwt

        print("[WARN] Password login worked but couldn't extract JWT.")
        print("       You can still use the session-based auth.")
        return None

    elif r.status_code == 401:
        body = r.json() if r.content else {}
        inquiry = body.get('inquiry', '')
        if inquiry:
            print(f"[ERR] WQ Brain requires Biometrics verification (inquiry={inquiry})")
            print("       Password-only login not available.")
            print("\n       You need to manually get the JWT from browser:")
        else:
            print(f"[ERR] Authentication failed: {r.text[:200]}")
        return None
    else:
        print(f"[ERR] Unexpected status: {r.status_code} - {r.text[:200]}")
        return None


def manual_jwt_input():
    """Let user paste JWT from browser DevTools"""
    print("\n" + "=" * 60)
    print("MANUAL JWT INPUT")
    print("=" * 60)
    print("""
How to get your JWT token:
  1. Open Chrome/Firefox, go to https://worldquantbrain.com and log in
  2. Press F12 → Application tab → Cookies → worldquantbrain.com
  3. Find the cookie named 't' and copy its Value
     (it should start with 'eyJ' and be ~200 characters)

Or from Network tab:
  1. F12 → Network tab → refresh the page
  2. Click any request → Headers → find 'Authorization: Bearer xxx'
  3. Copy the token after 'Bearer '
""")
    jwt = input("Paste JWT token (or press Enter to skip): ").strip()
    if jwt and jwt.startswith('eyJ'):
        return jwt
    if jwt:
        print("[WARN] Token doesn't look like a JWT, trying anyway...")
        return jwt
    return None


def main():
    print("=" * 55)
    print("WQ Brain Auth Refresher")
    print("=" * 55)

    email, password = load_credentials()
    if not email:
        sys.exit(1)

    print(f"Email: {email}")

    # Try password login
    jwt = try_password_login(email, password)

    # If password login failed (Biometrics required), ask for manual JWT
    if not jwt:
        jwt = manual_jwt_input()

    if not jwt:
        print("\n[FAIL] No JWT obtained. Cannot proceed.")
        print("Make sure you're logged into WQ Brain in your browser,")
        print("then re-run this script and paste the JWT token.")
        sys.exit(1)

    # Save to both locations
    save_jwt_auth_json(jwt, AUTH_FILE)
    save_jwt_auth_json(jwt, ROOT_AUTH)

    # Verify the token works
    print("\n[2] Verifying JWT...")
    sess = requests.Session()
    sess.headers['Authorization'] = f'Bearer {jwt}'
    r = sess.get(f"{API}/users/self", timeout=15)
    if r.status_code == 200:
        u = r.json()
        print(f"[OK] Token valid! User: {u.get('email')} | Level: {u.get('geniusLevel')}")
        print("\nNow run: python generation_two\\fetch_all_datafields.py")
    else:
        print(f"[FAIL] Token verification failed: HTTP {r.status_code}")
        print("The JWT may be invalid. Try getting a fresh one from browser.")


if __name__ == "__main__":
    main()
