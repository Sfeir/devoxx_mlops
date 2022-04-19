"""The setup script"""
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'pip>=21.3.1',
    'click>=7.1.2',
    'google-auth==2.3.3',
    'google-auth-oauthlib==0.4.6',
    'tensorflow>=2.8.0',
    'joblib>=1.1.0',
    'cloudml-hypertune'
]

setup(
    name='quickdraw_classifier',
    version='0.0.1',
    packages=find_packages(include=['quickdraw_classifier', 'quickdraw_classifier.*']),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.6",
    description="Quickdraw images classifier",
    zip_safe=False,
)
