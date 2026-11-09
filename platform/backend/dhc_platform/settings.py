import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(os.environ.get('DHC_PROJECT_ROOT', BASE_DIR.parent.parent)).resolve()
SECRET_KEY = os.environ.get('DHC_PLATFORM_SECRET_KEY', 'dhc-platform-dev-secret-key')
DEBUG = os.environ.get('DHC_PLATFORM_DEBUG', '1') == '1'
ALLOWED_HOSTS = [x.strip() for x in os.environ.get('DHC_PLATFORM_ALLOWED_HOSTS', '*').split(',') if x.strip()]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'dhc_platform.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'dhc_platform.wsgi.application'

if os.environ.get('MYSQL_DATABASE'):
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': os.environ.get('MYSQL_DATABASE', 'dhc_platform'),
            'USER': os.environ.get('MYSQL_USER', 'dhc'),
            'PASSWORD': os.environ.get('MYSQL_PASSWORD', 'dhcpass'),
            'HOST': os.environ.get('MYSQL_HOST', 'mysql'),
            'PORT': int(os.environ.get('MYSQL_PORT', '3306')),
            'OPTIONS': {
                'charset': 'utf8mb4',
                'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
            },
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = os.environ.get('TZ', 'Asia/Shanghai')
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

DHC_PLATFORM = {
    'REPO_ROOT': REPO_ROOT,
    'RUNTIME_ROOT': REPO_ROOT / 'platform' / 'runtime',
    'RESULTS_ROOT': REPO_ROOT / 'results',
    'DATA_ROOT': REPO_ROOT / 'data',
    'PYTHON_BIN': os.environ.get('DHC_PLATFORM_PYTHON_BIN', 'python'),
    'GITHUB_URL': os.environ.get('DHC_PLATFORM_GITHUB_URL', 'https://github.com/your-org/dyn_hetero_community'),
}
