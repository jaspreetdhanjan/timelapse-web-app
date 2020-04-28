import requests
from PIL import Image
from io import BytesIO


class BingSearch:
    SUBSCRIPTION_KEY = "ce85e3f76c3b4dec95d4044165dd5489"
    SEARCH_URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

    def search(self, search_term, n):
        """
        This implementation was sourced from:
        https://docs.microsoft.com/en-us/azure/cognitive-services/bing-image-search/quickstarts/python

        :param search_term: the search criteria of the Bing search
        :param n: the number of images to load
        :return: a list of URLs to the searched images
        """

        headers = {"Ocp-Apim-Subscription-Key": self.SUBSCRIPTION_KEY}

        params = {"q": search_term,
                  "license": "public",
                  "imageType": "photo",
                  "count": n}

        response = requests.get(self.SEARCH_URL, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        urls = [img["contentUrl"] for img in search_results["value"][:n]]

        return urls
