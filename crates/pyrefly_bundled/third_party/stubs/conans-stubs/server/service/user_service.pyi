from _typeshed import Incomplete
from conans.errors import AuthenticationException as AuthenticationException

class UserService:
    authenticator: Incomplete
    credentials_manager: Incomplete
    def __init__(self, authenticator, credentials_manager) -> None: ...
    def authenticate(self, username, password): ...
