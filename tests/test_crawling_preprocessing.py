"""Offline integration tests for shared crawling, CSV processing and inference."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from runtime.absa import inference
from runtime.chatbot import url_analyzer
from runtime.crawling import client, module1_product_list as m1, module2_product_detail as m2
from runtime.crawling import module3_product_reviews as m3
from runtime.preprocessing.clean_review import clean_review, clean_text, prepare_reviews
from runtime.preprocessing.main import preprocess_files


class CrawlingTests(unittest.TestCase):
    def test_batch_csvs_feed_preprocessor_and_keep_raw_reviews(self):
        raw = {'name': 'Sản phẩm A', 'categories': [{'name': 'Gia dụng'}],
               'short_description': '<p>Mô tả</p><script>hidden</script>', 'price': 123,
               'rating_average': 4.5, 'review_count': 1, 'brand': {'name': 'A'}}
        review = {'id': 7, 'content': 'SP ko tốt!!!', 'rating': 1,
                  'created_by': {'name': 'Khách'}, 'created_at': 1}
        with tempfile.TemporaryDirectory() as directory:
            raw_dir, processed = Path(directory) / 'raw', Path(directory) / 'processed'
            with patch.object(m1, 'fetch_product_page', return_value=[{'id': 123}, {'id': 123}]):
                products = m1.get_product_list(1846, 1, raw_dir, delay=0)
            with patch.object(m2, 'fetch_product_detail', return_value=raw):
                m2.get_product_detail(products, raw_dir, only_with_reviews=False, delay=0)
            with patch.object(m3, 'fetch_reviews_page', return_value=([review, review], 1)):
                m3.get_product_reviews(products, raw_dir, delay=0)
            preprocess_files(raw_dir, processed)
            with (processed / 'processed_products_reviews.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['content'], 'SP ko tốt!!!')
            self.assertEqual(rows[0]['clean_content'], 'sản phẩm không tốt')
            self.assertEqual(rows[0]['customer_name'], 'Khách')
            self.assertEqual(rows[0]['created_at'], '1970-01-01 00:00:01')
            details = pd.read_csv(processed / 'processed_details.csv')
            self.assertEqual(details.iloc[0]['short_description'], 'Mô tả')
            self.assertEqual(details.iloc[0]['category_name'], 'Gia dụng')
            self.assertIn('rating_average', details.columns)
            with patch.object(m1, 'fetch_product_page') as fetch:
                with self.assertRaises(FileExistsError):
                    m1.get_product_list(1846, 1, raw_dir)
                fetch.assert_not_called()
            with self.assertRaises(FileExistsError):
                preprocess_files(raw_dir, processed)

    def test_http_parameters_paging_fallback_and_errors(self):
        response = Mock(status_code=200)
        response.json.return_value = {'data': [], 'paging': {'total_pages': 3}}
        with patch('requests.get', return_value=response) as get:
            self.assertEqual(m3.fetch_reviews_page('123', 2), ([], 3))
        self.assertEqual(get.call_args.args[0], 'https://tiki.vn/api/v2/reviews')
        self.assertEqual(get.call_args.kwargs['params']['page'], 2)
        self.assertIn('sort', get.call_args.kwargs['params'])
        self.assertFalse(get.call_args.kwargs['allow_redirects'])
        response.json.return_value = []
        with patch('requests.get', return_value=response):
            with self.assertRaises(client.CrawlError):
                m2.fetch_product_detail('123')
        with patch('requests.get') as get:
            with self.assertRaises(ValueError):
                m2.fetch_product_detail('../reviews')
            get.assert_not_called()

    def test_unknown_paging_still_respects_cap_and_partial_failure_is_explicit(self):
        review = {'content': 'Tốt', 'id': 1}
        with patch.object(m3, 'fetch_reviews_page', return_value=([review], None)) as fetch:
            self.assertEqual(len(m3.crawl_reviews_for_product('123', max_pages=2, delay=0)), 1)
            self.assertEqual(fetch.call_count, 2)
        with patch.object(m3, 'fetch_reviews_page', side_effect=[([review], 3), client.CrawlError('HTTP 403')]):
            with self.assertRaises(client.CrawlError):
                m3.crawl_reviews_for_product('123', delay=0)

    def test_live_crawler_filters_after_normalization_preserves_original(self):
        reviews = [{'id': 1, 'content': 'SP ko tốt!!!'}, {'id': 2, 'content': 'sản phẩm không tốt'},
                   {'id': 3, 'content': '!!!'}]
        with patch.object(m3, 'fetch_reviews_page', return_value=(reviews, 1)):
            result = url_analyzer.crawl_reviews_fast('123')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['content'], 'SP ko tốt!!!')
        self.assertEqual(result[0]['clean_content'], 'sản phẩm không tốt')


class PreprocessingTests(unittest.TestCase):
    def test_normalization_handles_missing_values_and_keeps_negation(self):
        for value in (None, float('nan'), pd.NA, ''):
            self.assertEqual(clean_text(value), '')
        self.assertEqual(clean_text('SP ko tốt, ship chậm!!!'), 'sản phẩm không tốt giao hàng chậm')
        records = [{'product_id': '1', 'content': 'Tốt!'}, {'product_id': '1', 'content': 'tốt'},
                   {'product_id': '2', 'content': 'tốt'}, {'product_id': '1', 'content': '!!!'}]
        self.assertEqual(len(prepare_reviews(records)), 2)
        self.assertEqual(len(clean_review(pd.DataFrame(records))), 2)

    def test_inference_uses_normalized_text_without_refitting(self):
        predictor = Mock()
        predictor.predict.return_value = [{'quality': 'negative'}, {'quality': 'positive'}]
        with patch.object(inference, 'get_logreg', return_value=predictor):
            scores = inference.predict_and_aggregate([
                {'content': 'SP ko tốt!!!'}, {'content': '!!!'},
                {'content': 'typo', 'clean_content': 'đã sửa lỗi từ ngữ'},
            ])
        predictor.predict.assert_called_once_with(['sản phẩm không tốt', 'đã sửa lỗi từ ngữ'])
        self.assertEqual(scores['quality']['neg'], 1)
        self.assertEqual(scores['quality']['pos'], 1)
        predictor.fit.assert_not_called()

    def test_no_reviews_does_not_load_checkpoint(self):
        with patch.object(inference, 'get_logreg') as load:
            result = inference.predict_and_aggregate([{'content': '!!!'}])
        load.assert_not_called()
        self.assertEqual(result['quality']['total'], 0)


if __name__ == '__main__':
    unittest.main()
