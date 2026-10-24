"""HTTP boundary shared by all crawlers; never requests a user-supplied URL."""
import requests


class CrawlError(RuntimeError):
    """Tiki data could not be fetched or decoded."""


def get_json(path, params=None):
    try:
        response = requests.get(
            "https://tiki.vn/api/v2/" + path, params=params,
            timeout=(5, 20), allow_redirects=False,
            headers={"Accept": "application/json", "User-Agent": "TikiABSA/1.0"},
        )
        if response.status_code != 200:
            raise CrawlError(f"Tiki trả HTTP {response.status_code}; không thể lấy dữ liệu trực tiếp.")
        data = response.json()
        if not isinstance(data, dict):
            raise CrawlError("Dữ liệu Tiki trả về không đúng định dạng.")
        return data
    except (requests.RequestException, ValueError) as exc:
        raise CrawlError("Không đọc được dữ liệu Tiki; kiểm tra mạng hoặc thử lại sau.") from exc


def validate_product_id(value):
    text = str(value).strip()
    if not text.isascii() or not text.isdigit() or not 0 < int(text) <= 2**63 - 1:
        raise ValueError("product_id must be a positive integer")
    return str(int(text))
