# 这里定义api retriever (接入web)
from typing import List, Dict
import requests
import json

class SearchEngineRetriever:

    def __init__(
        self,
    ):
        self.serper_google_api_key = '0fb50c3b050c54c7b26181da42c2d8832e146f4e'
        self.serapi_bing_api_key = '73edf631bb5e6924c60ada1addb49fd6114f0048147c63ab7a90870f78dc4c8b'
        self.DEFAULT_SEARCH_ENGINE_TIMEOUT = 5
        
    def retrieve_from_google(
        self,
        query_text: str,
        max_hits_count: int = 3,
        document_type: str = "paragraph_text",
        corpus_name: str = None,
    ) -> List[Dict]:
        """
        Search with serper and return the contexts.
        """
        endpoint_url = "https://google.serper.dev/search"
        
        payload = json.dumps({
            "q": query_text,
            "num": (
                max_hits_count
                if max_hits_count % 10 == 0
                else (max_hits_count // 10 + 1) * 10
            ),
        })
        headers = {"X-API-KEY": self.serper_google_api_key, "Content-Type": "application/json"}
        
        response = requests.request(
            "POST",
            url=endpoint_url,
            headers=headers,
            data=payload,
            timeout=self.DEFAULT_SEARCH_ENGINE_TIMEOUT,
        )

        if not response.ok:
            raise "Search engine error."
            
        json_content = response.json()
        try:
            # convert to the same format as bing/google
            contexts = []
            contexts += [
                {"title": c["title"], "url": c["link"], "paragraph_text": c.get("snippet","")}
                for c in json_content["organic"]
            ]
            return contexts[:max_hits_count]
    
        except KeyError:
            return []
    
    def retrieve_from_bing(
        self,
        query_text: str,
        max_hits_count: int = 3,
        document_type: str = "paragraph_text",
        corpus_name: str = None,
    ) -> List[Dict]:
        """
        Search with serper and return the contexts.
        """
        import serpapi

        params = {
          "engine": "bing",
          "q": query_text,
          "cc": "US",
          "api_key": self.serapi_bing_api_key,
          "num": max_hits_count
          if max_hits_count % 10 == 0
          else (max_hits_count // 10 + 1) * 10,
          
        } # 看下检索数怎么设置
        
        search_results = serpapi.search(params)
        
        json_content = search_results #.json()
        
        try:
            # convert to the same format as bing/google
            contexts = []
            contexts += [
                {"title": c["title"], "url": c["link"], "paragraph_text": c.get("snippet","")}
                for c in json_content["organic_results"]
            ]
            return contexts[:max_hits_count]
    
        except KeyError:
            return []