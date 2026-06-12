from locust import HttpUser, task
import random

class FeatureUser(HttpUser):

    @task
    def query_feature(self):

        user_id = random.randint(1, 1000)

        self.client.get(
            f"/features/{user_id}"
        )