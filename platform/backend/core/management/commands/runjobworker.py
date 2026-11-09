from django.core.management.base import BaseCommand

from core.services.job_worker import run_worker_loop


class Command(BaseCommand):
    help = 'Run the DHC platform background worker.'

    def add_arguments(self, parser):
        parser.add_argument('--poll-sec', type=int, default=3)

    def handle(self, *args, **options):
        run_worker_loop(poll_sec=int(options['poll_sec']))
