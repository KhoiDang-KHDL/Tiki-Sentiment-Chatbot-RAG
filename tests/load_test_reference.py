"""
Production Load Test — Tiki RAG Chatbot
Queries generated from actual KB data (brands, categories, products)
Multi-scenario: normal, aspect, comparison, followup, edge case, stress
"""
from locust import HttpUser, task, between, events, tag
import random
import time
import logging

logger = logging.getLogger("loadtest")

# ── Queries from actual KB data ────────────────────────
# Product search queries
PRODUCT_QUERIES = [
    # By brand
    "sản phẩm Lock&Lock tốt nhất",
    "bình giữ nhiệt Elmich có tốt không",
    "đồ gia dụng Philips chất lượng",
    "sản phẩm LocknLock giá rẻ",
    "pin Energizer hay Duracell tốt hơn",
    "chảo Elmich Hera có bền không",
    "bình nước Kilner thủy tinh",
    "sản phẩm DandiHome đáng mua không",
    "nhang Nhang Xanh có thơm không",
    "cà phê Trung Nguyên Legend",
    # By category
    "bình giữ nhiệt dưới 500k",
    "bình đựng nước cho học sinh",
    "chảo chiên chống dính tốt nhất",
    "dụng cụ xay sinh tố cầm tay",
    "pin tiểu AA chính hãng giá rẻ",
    "pin sạc AA tốt nhất",
    "ống hút thủy tinh an toàn",
    "đồ nhà cửa đời sống tiện ích",
    # By specific product
    "bình giữ nhiệt Lock&Lock Energetic One-Touch",
    "cốc giữ nhiệt inox 304 Elmich EL8345",
    "bình giữ nhiệt Slo 2in1 LocknLock",
    "chảo chống dính Elmich Hera II",
    "chảo chống dính vân đá Elmich",
    "bộ cây lau nhà Lock&Lock Compact Spin Mop",
    "bình nước cao cấp Biwa Plus",
    "bóng đèn Philips LED Bulb",
    "cầu dao Schneider Electric",
    # By price range
    "sản phẩm dưới 100k",
    "đồ gia dụng dưới 300k",
    "bình giữ nhiệt dưới 200k",
    "sản phẩm từ 200k đến 500k",
    "đồ nhà bếp cao cấp trên 500k",
]

# Aspect-focused queries
ASPECT_QUERIES = [
    "sản phẩm nào giao hàng nhanh nhất",
    "sản phẩm được đánh giá tốt về chất lượng",
    "sản phẩm nào đóng gói cẩn thận",
    "sản phẩm dịch vụ hậu mãi tốt",
    "đồ gia dụng giá rẻ mà chất lượng ổn",
    "sản phẩm nào nhiều người khen về giá",
    "bình giữ nhiệt nào giữ nóng lâu nhất",
    "sản phẩm nào ít bị phàn nàn giao hàng",
    "đồ dùng nhà bếp bền nhất theo đánh giá",
    "sản phẩm nào mô tả đúng với thực tế",
    "Lock&Lock sản phẩm nào chất lượng nhất",
    "Elmich sản phẩm nào đáng mua nhất",
    "sản phẩm nào đóng gói đẹp nhất",
    "sản phẩm nào giao hàng chậm nên tránh",
    "chảo chống dính nào bền nhất",
]

# Comparison queries
COMPARISON_QUERIES = [
    "so sánh cốc giữ nhiệt Lock&Lock và Elmich",
    "nên mua Elmich hay Lock&Lock",
    "bình giữ nhiệt nào tốt nhất dưới 500k",
    "pin Energizer hay Duracell bền hơn",
    "chảo Elmich Hera II hay chảo vân đá tốt hơn",
    "so sánh bình nước thủy tinh Kilner với bình inox",
    "lau nhà Lock&Lock hay lau nhà thường tốt hơn",
]

# Follow-up queries (dùng cho multi-turn)
FOLLOWUP_QUERIES = [
    "loại nào rẻ nhất",
    "có loại nào tốt hơn không",
    "giao hàng có nhanh không",
    "cái nào được đánh giá cao nhất",
    "có màu khác không",
    "dung tích bao nhiêu",
    "bảo hành bao lâu",
    "mua ở đâu rẻ hơn",
    "có combo giảm giá không",
    "loại nào phù hợp cho trẻ em",
]

# Edge cases
EDGE_CASE_QUERIES = [
    "coc giu nhiet",
    "binh nuoc",
    "lock&lock",
    "pin tieu",
    "chao chong dinh",
    "do gia dung",
    "???",
    "a",
    " ",
    "sản phẩm tốt nhất trên tiki là gì vậy bạn ơi mình đang cần mua gấp một cái bình giữ nhiệt để mang đi làm hàng ngày cho tiện",
    "bình giữ nhiệt 🔥 tốt nhất",
    "tôi cần mua quà tặng cho mẹ, gợi ý giúp",
    "có gì hot không",
    "hello",
    "sản phẩm",
]

ALL_QUERIES = PRODUCT_QUERIES + ASPECT_QUERIES + COMPARISON_QUERIES + EDGE_CASE_QUERIES


class NormalUser(HttpUser):
    """User bình thường — 50% traffic."""
    weight = 5
    wait_time = between(3, 8)

    @task(5)
    @tag("chat", "product")
    def ask_product(self):
        query = random.choice(PRODUCT_QUERIES)
        with self.client.post(
            "/chat",
            json={"query": query, "top_k": 5},
            name="/chat [product]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if data.get("answer") and data.get("sources"):
                        resp.success()
                    else:
                        resp.failure("Empty answer or sources")
                except Exception:
                    resp.failure("Invalid JSON")
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(3)
    @tag("chat", "aspect")
    def ask_aspect(self):
        query = random.choice(ASPECT_QUERIES)
        with self.client.post(
            "/chat",
            json={"query": query, "top_k": 5},
            name="/chat [aspect]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()

    @task(2)
    @tag("chat", "comparison")
    def ask_comparison(self):
        query = random.choice(COMPARISON_QUERIES)
        with self.client.post(
            "/chat",
            json={"query": query, "top_k": 5},
            name="/chat [comparison]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()

    @task(1)
    @tag("health")
    def health_check(self):
        self.client.get("/health", name="/health")


class PowerUser(HttpUser):
    """Multi-turn user — 20% traffic."""
    weight = 2
    wait_time = between(2, 5)

    @task
    @tag("chat", "multi-turn")
    def multi_turn_session(self):
        query1 = random.choice(PRODUCT_QUERIES)
        self.client.post(
            "/chat",
            json={"query": query1, "top_k": 5},
            name="/chat [multi-turn-1]",
        )
        time.sleep(random.uniform(2, 5))

        query2 = random.choice(FOLLOWUP_QUERIES)
        self.client.post(
            "/chat",
            json={"query": query2, "top_k": 5},
            name="/chat [multi-turn-2]",
        )


class StressUser(HttpUser):
    """Stress test — gửi liên tục — 20% traffic."""
    weight = 2
    wait_time = between(0.5, 1.5)

    @task
    @tag("chat", "stress")
    def rapid_fire(self):
        query = random.choice(ALL_QUERIES)
        with self.client.post(
            "/chat",
            json={"query": query, "top_k": 5},
            name="/chat [stress]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                resp.failure("Rate limited")
            elif resp.status_code == 503:
                resp.failure("Server overloaded")
            else:
                resp.failure(f"Status {resp.status_code}")


class HealthMonitor(HttpUser):
    """Health monitor — 10% traffic."""
    weight = 1
    wait_time = between(1, 2)

    @task
    @tag("health")
    def check_health(self):
        with self.client.get(
            "/health",
            name="/health [monitor]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok":
                    resp.success()
                else:
                    resp.failure("Unhealthy")


class SearchOnlyUser(HttpUser):
    """Test embedding + Qdrant only, no LLM."""
    weight = 10
    wait_time = between(1, 3)

    @task
    @tag("search")
    def search_product(self):
        query = random.choice(ALL_QUERIES)
        with self.client.post(
            "/search",
            json={"query": query, "top_k": 5},
            name="/search [no-LLM]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("=== Load test STARTED ===")
    logger.info(f"Target: {environment.host}")
    logger.info(f"Total unique queries: {len(ALL_QUERIES)}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("=== Load test STOPPED ===")
