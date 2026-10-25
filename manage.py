#!/usr/bin/env python
"""Django command-line utility for the Upscaler (development and checks).

The desktop app starts its own embedded server; this is only needed for
``python manage.py check`` or to open the Upscaler page in a browser while
developing (``python manage.py runserver 127.0.0.1:8000``). Without the
desktop shell there are no native dialogs, so the page asks for typed paths.
"""

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.web.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
