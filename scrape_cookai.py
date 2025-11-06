#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cookai-recipe.com からCookpadレシピをスクレイピングするスクリプト

機能:
1. カテゴリページを再帰的にスクレイピング
2. 記事ページからCookpadリンクを取得
3. Cookpadページから情報を抽出（タイトル、画像URL、作者等）
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote
import re
import time
import json
from typing import List, Dict, Set
import sys

class CookAIScraper:
    def __init__(self, delay: float = 1.0, insert_file: str = 'insert_statements.sql'):
        """
        初期化
        
        Args:
            delay: リクエスト間の待機時間（秒）
            insert_file: INSERT文を保存するファイル名
        """
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.visited_urls: Set[str] = set()
        self.visited_article_urls: Set[str] = set()  # 訪問済み記事ページURL
        self.visited_cookpad_urls: Set[str] = set()  # 訪問済みCookpadページURL
        self.cookpad_urls: Set[str] = set()
        self.results: List[Dict] = []
        self.insert_file = insert_file
        # INSERT文ファイルを初期化（既存の内容は保持）
        
    def get_page(self, url: str) -> tuple:
        """
        ページを取得してパース
        
        Returns:
            (BeautifulSoup, 最終URL) のタプル。エラー時は (None, None)
        """
        if url in self.visited_urls:
            return None, None
            
        try:
            print(f"取得中: {url}")
            response = requests.get(url, headers=self.headers, timeout=10, allow_redirects=True)
            response.raise_for_status()
            final_url = response.url  # リダイレクト先の最終URL
            self.visited_urls.add(url)
            time.sleep(self.delay)
            return BeautifulSoup(response.content, 'html.parser'), final_url
        except Exception as e:
            print(f"エラー ({url}): {e}")
            return None, None
    
    def get_article_urls_from_category(self, category_url: str) -> Set[str]:
        """カテゴリページから記事URLを取得"""
        soup, _ = self.get_page(category_url)
        if not soup:
            return set()
        
        article_urls = set()
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href', '')
            full_url = urljoin(category_url, href)
            
            # 記事ページのURLパターン
            if (re.search(r'/[\w-]+/\d+\.html', href) or 
                (href.startswith('/') and href.endswith('.html') and href.count('/') >= 2)):
                article_urls.add(full_url)
        
        return article_urls
    
    def get_pagination_urls(self, category_url: str) -> Set[str]:
        """カテゴリページからページネーションURLを取得"""
        soup, _ = self.get_page(category_url)
        if not soup:
            return set()
        
        pagination_urls = set()
        all_links = soup.find_all('a', href=True)
        
        base_category = category_url.split('?')[0]  # クエリパラメータを除去
        
        for link in all_links:
            href = link.get('href', '')
            full_url = urljoin(category_url, href)
            
            # ページネーションリンク（例: /category/xxx/page/2）
            if '/page/' in href and 'cookai-recipe.com' in full_url:
                pagination_urls.add(full_url)
            # 同じカテゴリのリンクで、ページ番号が含まれている場合
            elif base_category in full_url and ('page' in href.lower() or re.search(r'page[/=]\d+', href)):
                pagination_urls.add(full_url)
        
        return pagination_urls
    
    def get_category_urls(self, base_url: str = "https://cookai-recipe.com") -> Set[str]:
        """
        カテゴリページのURLを取得
        
        Args:
            base_url: ベースURL（デフォルトはトップページ）
            
        Returns:
            カテゴリページのURLセット
        """
        soup, _ = self.get_page(base_url)
        if not soup:
            return set()
        
        category_urls = set()
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href', '')
            full_url = urljoin(base_url, href)
            
            # カテゴリページのURLパターン: /category/...
            # より柔軟に検索（相対パスと絶対パスの両方に対応）
            if '/category/' in href:
                # ページネーションURLは除外
                if '/page/' not in href:
                    # cookai-recipe.comが含まれているか、または相対パスの場合
                    if 'cookai-recipe.com' in full_url or href.startswith('/category/'):
                        category_urls.add(full_url)
        
        return category_urls
    
    def get_tag_urls(self, base_url: str = "https://cookai-recipe.com", category_urls: Set[str] = None) -> Set[str]:
        """
        タグページのURLを取得
        
        Args:
            base_url: ベースURL（デフォルトはトップページ）
            category_urls: カテゴリURLのセット（指定された場合、カテゴリページからもタグを取得）
            
        Returns:
            タグページのURLセット
        """
        tag_urls = set()
        
        # トップページからタグを取得
        soup, _ = self.get_page(base_url)
        if soup:
            all_links = soup.find_all('a', href=True)
            
            for link in all_links:
                href = link.get('href', '')
                full_url = urljoin(base_url, href)
                
                # タグページのURLパターン: /tag/...
                if '/tag/' in href:
                    # ページネーションURLは除外
                    if '/page/' not in href:
                        # cookai-recipe.comが含まれているか、または相対パスの場合
                        if 'cookai-recipe.com' in full_url or href.startswith('/tag/'):
                            tag_urls.add(full_url)
        
        # カテゴリページからもタグを取得（オプション）
        if category_urls:
            print(f"  カテゴリページからタグを取得中...")
            for cat_url in list(category_urls)[:5]:  # 最初の5つのカテゴリページから取得
                cat_soup, _ = self.get_page(cat_url)
                if cat_soup:
                    cat_links = cat_soup.find_all('a', href=True)
                    for link in cat_links:
                        href = link.get('href', '')
                        full_url = urljoin(cat_url, href)
                        
                        if '/tag/' in href and '/page/' not in href:
                            if 'cookai-recipe.com' in full_url or href.startswith('/tag/'):
                                tag_urls.add(full_url)
        
        return tag_urls
    
    def get_cookpad_urls_from_article(self, article_url: str) -> Set[str]:
        """記事ページからCookpadリンクを取得"""
        # 訪問済みチェック
        if article_url in self.visited_article_urls:
            print(f"  [スキップ] 既に訪問済み: {article_url}")
            return set()
        
        soup, _ = self.get_page(article_url)
        if not soup:
            return set()
        
        # 訪問済みとしてマーク
        self.visited_article_urls.add(article_url)
        
        cookpad_urls = set()
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href', '')
            if 'cookpad.com/recipe/' in href:
                full_url = urljoin(article_url, href)
                # URLを正規化（クエリパラメータやフラグメントを除去）
                parsed = urlparse(full_url)
                normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                cookpad_urls.add(normalized)
        
        return cookpad_urls
    
    def scrape_cookpad_page(self, cookpad_url: str) -> Dict:
        """Cookpadページから情報を抽出"""
        # 訪問済みチェック
        if cookpad_url in self.visited_cookpad_urls:
            print(f"  [スキップ] 既に訪問済み: {cookpad_url}")
            return None
        
        soup, final_url = self.get_page(cookpad_url)
        if not soup:
            return None
        
        # 訪問済みとしてマーク
        self.visited_cookpad_urls.add(cookpad_url)
        
        # リダイレクト先のURLを使用（リダイレクトされていない場合は元のURLを使用）
        actual_url = final_url if final_url else cookpad_url
        
        # リダイレクト先のURLも訪問済みとしてマーク（異なる場合）
        if final_url and final_url != cookpad_url:
            self.visited_cookpad_urls.add(final_url)
        
        result = {
            'url': actual_url,  # リダイレクト先のURLを使用
            'title': None,
            'image_url': None,
            'author': None,
            'recipe_id': None,
            'tsukurepo_count': None
        }
        
        # タイトル
        title_tag = soup.find('title')
        if title_tag:
            result['title'] = title_tag.text.strip()
        
        # h1タグからタイトルを取得（より正確な場合がある）
        h1_tag = soup.find('h1')
        if h1_tag:
            h1_text = h1_tag.text.strip()
            if h1_text and len(h1_text) > 0:
                result['title'] = h1_text
        
        # 画像URL
        # alt属性に「レシピのメイン写真」や「レシピ-メイン写真」が含まれるimgタグを探す
        main_images = soup.find_all('img', alt=True)
        for img in main_images:
            alt_text = img.get('alt', '')
            if 'レシピ' in alt_text and ('メイン写真' in alt_text or 'メイン-写真' in alt_text):
                src = img.get('src', '')
                if src and src.startswith('http'):
                    result['image_url'] = src
                    break
        
        # 見つからない場合はog:imageを試す
        if not result['image_url']:
            og_image = soup.find('meta', property='og:image')
            if og_image:
                result['image_url'] = og_image.get('content', '')
        
        # 作者情報
        # href="/jp/users/..." を持つaタグを探す
        user_links = soup.find_all('a', href=re.compile(r'/jp/users/\d+'))
        for user_link in user_links:
            # その中でfont-semiboldクラスを持つspanタグを探す
            author_span = user_link.find('span', class_=lambda x: x and 'font-semibold' in x)
            if author_span:
                author_text = author_span.text.strip()
                if author_text and len(author_text) > 0:
                    result['author'] = author_text
                    break
        
        # レシピIDをリダイレクト先のURLから抽出
        # 新形式: /jp/recipes/22635859
        match = re.search(r'/jp/recipes/(\d+)', actual_url)
        if match:
            result['recipe_id'] = match.group(1)
        else:
            # 旧形式の場合: /recipe/723956
            match = re.search(r'/recipe/(\d+)', actual_url)
            if match:
                result['recipe_id'] = match.group(1)
        
        # つくれぽ数
        # 方法1: JSON-LDのcommentCountから取得
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and 'commentCount' in data:
                    result['tsukurepo_count'] = int(data['commentCount'])
                    break
            except (json.JSONDecodeError, ValueError):
                continue
        
        # 方法2: data-cooksnapped-count-cooksnaps-count-value属性から取得
        if result['tsukurepo_count'] is None:
            cooksnapped_div = soup.find('div', attrs={'data-cooksnapped-count-cooksnaps-count-value': True})
            if cooksnapped_div:
                count_value = cooksnapped_div.get('data-cooksnapped-count-cooksnaps-count-value')
                if count_value:
                    try:
                        result['tsukurepo_count'] = int(count_value)
                    except ValueError:
                        pass
        
        return result
    
    def escape_sql_string(self, value: str) -> str:
        """SQL文字列のエスケープ（シングルクォートをエスケープ）"""
        if value is None:
            return ''
        return str(value).replace("'", "''")
    
    def generate_insert_statement(self, result: Dict) -> str:
        """
        INSERT文を生成
        
        Args:
            result: スクレイピング結果の辞書
            
        Returns:
            INSERT文の文字列（つくれぽ数が500未満の場合は空文字列）
        """
        # つくれぽ数が500未満の場合はスキップ
        tsukurepo_count = result.get('tsukurepo_count')
        if tsukurepo_count is None or tsukurepo_count < 500:
            return ''
        
        # 必須フィールドの取得
        recipe_id = result.get('recipe_id')
        image_url = result.get('image_url', '')
        title = result.get('title', '')
        author = result.get('author', '')
        
        # recipe_idが無い場合はスキップ
        if not recipe_id:
            return ''
        
        # タイトルに「 by 作者名」を追加
        if author:
            title = f"{title} by {author}"
        
        # 値をエスケープ
        escaped_image = self.escape_sql_string(image_url)
        escaped_title = self.escape_sql_string(title)
        
        # INSERT文を生成
        insert_sql = f"insert into repo values ( 'sysop', {recipe_id}, '{escaped_image}', '{escaped_title}', 0, {tsukurepo_count}, '', '', 0, 0);"
        
        return insert_sql
    
    def append_insert_statement(self, result: Dict):
        """
        INSERT文をファイルに追記
        
        Args:
            result: スクレイピング結果の辞書
        """
        insert_sql = self.generate_insert_statement(result)
        if insert_sql:
            with open(self.insert_file, 'a', encoding='utf-8') as f:
                f.write(insert_sql + '\n')
            print(f"    -> INSERT文を {self.insert_file} に追加しました")
    
    def scrape_category(self, category_url: str, max_pages: int = 10):
        """
        カテゴリページを再帰的にスクレイピング
        
        Args:
            category_url: カテゴリページのURL
            max_pages: 最大ページ数（ページネーション）
        """
        print(f"\n{'='*60}")
        print(f"カテゴリページをスクレイピング開始: {category_url}")
        print(f"{'='*60}")
        
        # ページネーションURLを取得
        pagination_urls = self.get_pagination_urls(category_url)
        all_category_urls = {category_url} | pagination_urls
        
        # 最大ページ数に制限
        category_urls_list = list(all_category_urls)[:max_pages]
        print(f"処理するカテゴリページ数: {len(category_urls_list)}")
        
        # 各カテゴリページから記事URLを取得
        all_article_urls = set()
        for cat_url in category_urls_list:
            article_urls = self.get_article_urls_from_category(cat_url)
            all_article_urls.update(article_urls)
            print(f"  {cat_url}: {len(article_urls)}件の記事が見つかりました")
        
        print(f"\n合計記事数: {len(all_article_urls)}")
        
        # 訪問済みの記事URLを除外
        unvisited_article_urls = all_article_urls - self.visited_article_urls
        skipped_count = len(all_article_urls) - len(unvisited_article_urls)
        if skipped_count > 0:
            print(f"訪問済み記事数（スキップ）: {skipped_count}")
        print(f"処理対象記事数: {len(unvisited_article_urls)}")
        
        # 各記事ページからCookpadリンクを取得
        print(f"\n記事ページからCookpadリンクを取得中...")
        for i, article_url in enumerate(unvisited_article_urls, 1):
            print(f"  [{i}/{len(unvisited_article_urls)}] {article_url}")
            cookpad_urls = self.get_cookpad_urls_from_article(article_url)
            self.cookpad_urls.update(cookpad_urls)
            print(f"    -> {len(cookpad_urls)}件のCookpadリンクが見つかりました")
        
        print(f"\n合計Cookpadリンク数: {len(self.cookpad_urls)}")
        
        # 訪問済みのCookpadURLを除外
        unvisited_cookpad_urls = self.cookpad_urls - self.visited_cookpad_urls
        skipped_cookpad_count = len(self.cookpad_urls) - len(unvisited_cookpad_urls)
        if skipped_cookpad_count > 0:
            print(f"訪問済みCookpadリンク数（スキップ）: {skipped_cookpad_count}")
        print(f"処理対象Cookpadリンク数: {len(unvisited_cookpad_urls)}")
        
        # Cookpadページをスクレイピング
        print(f"\nCookpadページをスクレイピング中...")
        for i, cookpad_url in enumerate(unvisited_cookpad_urls, 1):
            print(f"  [{i}/{len(unvisited_cookpad_urls)}] {cookpad_url}")
            result = self.scrape_cookpad_page(cookpad_url)
            if result:
                self.results.append(result)
                tsukurepo = result.get('tsukurepo_count', 0)
                print(f"    -> タイトル: {result.get('title', 'N/A')[:50]}")
                print(f"    -> つくれぽ数: {tsukurepo}")
                # INSERT文を生成してファイルに追記
                self.append_insert_statement(result)
        
        return self.results
    
    def scrape_tag(self, tag_url: str, max_pages: int = 10):
        """
        タグページを再帰的にスクレイピング
        
        Args:
            tag_url: タグページのURL
            max_pages: 最大ページ数（ページネーション）
        """
        print(f"\n{'='*60}")
        print(f"タグページをスクレイピング開始: {tag_url}")
        print(f"{'='*60}")
        
        # ページネーションURLを取得
        pagination_urls = self.get_pagination_urls(tag_url)
        all_tag_urls = {tag_url} | pagination_urls
        
        # 最大ページ数に制限
        tag_urls_list = list(all_tag_urls)[:max_pages]
        print(f"処理するタグページ数: {len(tag_urls_list)}")
        
        # 各タグページから記事URLを取得
        all_article_urls = set()
        for tg_url in tag_urls_list:
            article_urls = self.get_article_urls_from_category(tg_url)
            all_article_urls.update(article_urls)
            print(f"  {tg_url}: {len(article_urls)}件の記事が見つかりました")
        
        print(f"\n合計記事数: {len(all_article_urls)}")
        
        # 訪問済みの記事URLを除外
        unvisited_article_urls = all_article_urls - self.visited_article_urls
        skipped_count = len(all_article_urls) - len(unvisited_article_urls)
        if skipped_count > 0:
            print(f"訪問済み記事数（スキップ）: {skipped_count}")
        print(f"処理対象記事数: {len(unvisited_article_urls)}")
        
        # 各記事ページからCookpadリンクを取得
        print(f"\n記事ページからCookpadリンクを取得中...")
        for i, article_url in enumerate(unvisited_article_urls, 1):
            print(f"  [{i}/{len(unvisited_article_urls)}] {article_url}")
            cookpad_urls = self.get_cookpad_urls_from_article(article_url)
            self.cookpad_urls.update(cookpad_urls)
            print(f"    -> {len(cookpad_urls)}件のCookpadリンクが見つかりました")
        
        print(f"\n合計Cookpadリンク数: {len(self.cookpad_urls)}")
        
        # 訪問済みのCookpadURLを除外
        unvisited_cookpad_urls = self.cookpad_urls - self.visited_cookpad_urls
        skipped_cookpad_count = len(self.cookpad_urls) - len(unvisited_cookpad_urls)
        if skipped_cookpad_count > 0:
            print(f"訪問済みCookpadリンク数（スキップ）: {skipped_cookpad_count}")
        print(f"処理対象Cookpadリンク数: {len(unvisited_cookpad_urls)}")
        
        # Cookpadページをスクレイピング
        print(f"\nCookpadページをスクレイピング中...")
        for i, cookpad_url in enumerate(unvisited_cookpad_urls, 1):
            print(f"  [{i}/{len(unvisited_cookpad_urls)}] {cookpad_url}")
            result = self.scrape_cookpad_page(cookpad_url)
            if result:
                self.results.append(result)
                tsukurepo = result.get('tsukurepo_count', 0)
                print(f"    -> タイトル: {result.get('title', 'N/A')[:50]}")
                print(f"    -> つくれぽ数: {tsukurepo}")
                # INSERT文を生成してファイルに追記
                self.append_insert_statement(result)
        
        return self.results
    
    def scrape_all_categories(self, category_urls: List[str], max_pages: int = 10):
        """
        複数のカテゴリページを順番にスクレイピング
        
        Args:
            category_urls: カテゴリページのURLリスト
            max_pages: 各カテゴリの最大ページ数（ページネーション）
        """
        print(f"\n{'='*80}")
        print(f"複数カテゴリのスクレイピングを開始")
        print(f"処理対象カテゴリ数: {len(category_urls)}")
        print(f"{'='*80}")
        
        for idx, category_url in enumerate(category_urls, 1):
            print(f"\n{'#'*80}")
            print(f"カテゴリ {idx}/{len(category_urls)}: {category_url}")
            print(f"{'#'*80}")
            
            # 1カテゴリの処理を実行（記事ページ解析⇒Cookpadページスクレイピング）
            self.scrape_category(category_url, max_pages=max_pages)
            
            print(f"\nカテゴリ {idx}/{len(category_urls)} の処理が完了しました")
            print(f"現在の累計結果数: {len(self.results)}")
        
        print(f"\n{'='*80}")
        print(f"全カテゴリのスクレイピングが完了しました")
        print(f"{'='*80}")
        
        return self.results
    
    def scrape_all_tags(self, tag_urls: List[str], max_pages: int = 10):
        """
        複数のタグページを順番にスクレイピング
        
        Args:
            tag_urls: タグページのURLリスト
            max_pages: 各タグの最大ページ数（ページネーション）
        """
        print(f"\n{'='*80}")
        print(f"複数タグのスクレイピングを開始")
        print(f"処理対象タグ数: {len(tag_urls)}")
        print(f"{'='*80}")
        
        for idx, tag_url in enumerate(tag_urls, 1):
            print(f"\n{'#'*80}")
            print(f"タグ {idx}/{len(tag_urls)}: {tag_url}")
            print(f"{'#'*80}")
            
            # 1タグの処理を実行（記事ページ解析⇒Cookpadページスクレイピング）
            self.scrape_tag(tag_url, max_pages=max_pages)
            
            print(f"\nタグ {idx}/{len(tag_urls)} の処理が完了しました")
            print(f"現在の累計結果数: {len(self.results)}")
        
        print(f"\n{'='*80}")
        print(f"全タグのスクレイピングが完了しました")
        print(f"{'='*80}")
        
        return self.results
    
    def save_results(self, output_file: str = 'cookpad_recipes.json'):
        """結果をJSONファイルに保存"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n結果を {output_file} に保存しました（{len(self.results)}件）")


def main():
    """メイン処理"""
    # 最大ページ数の指定（オプション）
    max_pages = 10
    if len(sys.argv) > 1 and sys.argv[-1].isdigit():
        max_pages = int(sys.argv[-1])
    
    scraper = CookAIScraper(delay=1.0)  # 1秒待機
    
    # トップページからカテゴリとタグのURLを取得
    top_url = "https://cookai-recipe.com/"
    print(f"\n{'='*80}")
    print(f"トップページからカテゴリとタグを取得中: {top_url}")
    print(f"{'='*80}")
    
    category_urls = list(scraper.get_category_urls(top_url))
    print(f"\n取得したカテゴリ数: {len(category_urls)}")
    
    if category_urls:
        print(f"\nカテゴリ一覧:")
        for cat_url in category_urls:
            print(f"  - {cat_url}")
    
    # タグを取得（カテゴリページからも取得を試みる）
    tag_urls = list(scraper.get_tag_urls(top_url, set(category_urls)))
    print(f"\n取得したタグ数: {len(tag_urls)}")
    
    if tag_urls:
        print(f"\nタグ一覧:")
        for tag_url in tag_urls:
            print(f"  - {tag_url}")
    else:
        print(f"\n警告: タグが見つかりませんでした。サイト構造を確認してください。")
    
    # カテゴリを順番にスクレイピング
    if category_urls:
        print(f"\n{'='*80}")
        print(f"カテゴリのスクレイピングを開始")
        print(f"{'='*80}")
        scraper.scrape_all_categories(category_urls, max_pages=max_pages)
    
    # タグを順番にスクレイピング
    if tag_urls:
        print(f"\n{'='*80}")
        print(f"タグのスクレイピングを開始")
        print(f"{'='*80}")
        scraper.scrape_all_tags(tag_urls, max_pages=max_pages)
    
    # 結果を保存
    scraper.save_results('cookpad_recipes.json')
    
    # サマリー表示
    print(f"\n{'='*60}")
    print("スクレイピング完了")
    print(f"{'='*60}")
    print(f"訪問したURL数: {len(scraper.visited_urls)}")
    print(f"訪問した記事ページ数: {len(scraper.visited_article_urls)}")
    print(f"訪問したCookpadページ数: {len(scraper.visited_cookpad_urls)}")
    print(f"取得したCookpadリンク数: {len(scraper.cookpad_urls)}")
    print(f"成功したスクレイピング数: {len(scraper.results)}")
    
    if scraper.results:
        print(f"\n最初の10件の結果:")
        for i, result in enumerate(scraper.results[:10], 1):
            print(f"\n{i}. {result.get('title', 'N/A')}")
            print(f"   画像: {result.get('image_url', 'N/A')}")
            print(f"   作者: {result.get('author', 'N/A')}")
            tsukurepo = result.get('tsukurepo_count')
            if tsukurepo is not None:
                print(f"   つくれぽ数: {tsukurepo}")
            else:
                print(f"   つくれぽ数: N/A")
            print(f"   URL: {result.get('url', 'N/A')}")


if __name__ == "__main__":
    main()

