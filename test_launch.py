import io
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import launch


class LauncherTests(unittest.TestCase):
    def test_reuses_running_app_without_starting_second_server(self):
        page=io.BytesIO('<title>镜头索引 · 本地素材搜索</title>'.encode())
        with patch('launch.urlopen',return_value=page), patch('launch.webbrowser.open') as browser, patch('launch.app.main') as server:
            launch.main()
            browser.assert_called_once_with(launch.URL)
            server.assert_not_called()

    def test_starts_server_when_not_running(self):
        with patch('launch.urlopen',side_effect=URLError('connection refused')), patch('launch.app.main') as server:
            launch.main()
            server.assert_called_once_with(open_browser=True)

    def test_does_not_reuse_unrelated_page(self):
        with patch('launch.urlopen',return_value=io.BytesIO(b'<title>Other app</title>')), patch('launch.webbrowser.open') as browser:
            with self.assertRaises(RuntimeError):
                launch.main()
            browser.assert_not_called()

    def test_unrelated_http_error_does_not_start_second_server(self):
        with patch('launch.urlopen',side_effect=HTTPError(launch.URL,404,'not found',{},None)), patch('launch.app.main') as server:
            with self.assertRaises(RuntimeError):
                launch.main()
            server.assert_not_called()


if __name__ == '__main__':
    unittest.main()
