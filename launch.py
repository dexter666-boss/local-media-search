"""Start the local UI, or reuse this application's existing local server."""
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import webbrowser

import app

URL = 'http://127.0.0.1:8765/'


def main():
    try:
        with urlopen(URL, timeout=2) as response:
            page = response.read().decode('utf-8', errors='replace')
        if '<title>镜头索引 · 本地素材搜索</title>' not in page:
            raise RuntimeError('8765 端口正在运行其他服务，请先解决端口冲突。')
    except HTTPError as error:
        raise RuntimeError('8765 端口上的服务不是当前工具，请检查端口冲突。') from error
    except URLError:
        app.main(open_browser=True)
    else:
        print('Local Media Search is already running. Opening the browser.', flush=True)
        webbrowser.open(URL)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f'Launch failed: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
