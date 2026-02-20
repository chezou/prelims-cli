from prelims.processor.base import BaseFrontMatterProcessor

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class EmbeddingRecommender(BaseFrontMatterProcessor):
    def __init__(self, db_path, model_name, collection_name, permalink_base='', topk=3, lower_path=True, init_run=False):
        self.permalink_base = permalink_base
        self.topk = topk
        self.lower_path = lower_path
        self.init_run = init_run

        self.client = PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

    def process(self, posts, allow_overwrite=True):
        contents = [post.content for post in posts]
        paths = [str(post.path) for post in posts]
        titles = [post.title for post in posts]

        results = []
        if self.init_run:
            collection = self.create_collection(contents, paths)
            results = collection.query(query_texts=titles, n_results=self.topk + 1)
        else:
            collection = self.client.get_collection(self.collection_name, embedding_function=self.embedding_function)
        
        for post in posts:
            if not allow_overwrite and post.front_matter.recommendations:
                continue

            related_articles = collection.query(query_texts=[post.title], n_results=self.topk + 1)


    def create_collection(self, contents, paths):
        collection = self.client.create_collection(self.collection_name, embedding_function=self.embedding_function)
        collection.add(documents=contents, ids=paths)
        return collection
