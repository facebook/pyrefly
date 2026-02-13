from conans.model.rest_routes import RestRoutes as RestRoutes

class BottleRoutes(RestRoutes):
    def __getattribute__(self, item): ...
