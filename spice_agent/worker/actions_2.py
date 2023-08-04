import asyncio
import json
import logging
import os
import ssl
import sys

import aio_pika
import aio_pika.abc

from spice_agent.utils.config import (
    SPICE_ROUND_VERIFICATION_FILEPATH,
    SPICE_TRAINING_FILEPATH,
    read_config_file,
)
from spice_agent.utils.version import get_current_version

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

LOGGER = logging.getLogger(__name__)


class AsyncWorker:
    def __init__(self, spice, loop) -> None:
        self.spice = spice
        self.device = self.spice.get_device()
        self.channel = None
        self.loop = loop

    async def _claim_message_callback(self, message, data):
        """
        Based on if we get a run_id from the message consume
        """
        inference_job_id = data.get("inference_job_id", None)
        LOGGER.info(f" [*] Processing message for {inference_job_id}")

        if inference_job_id:
            # ack message at inference level so another machine does not steal the
            # message while inference is running
            await message.ack()
            await self.spice.inference._async_update_inference_job(
                inference_job_id=inference_job_id, status="CLAIMED"
            )

            await self.spice.inference.run_pipeline(inference_job_id=inference_job_id)

            # while True:
            #     await asyncio.sleep(0.0)
            #     LOGGER.info(" [*] hello world.")
        LOGGER.info(" [*] Completed inference job.")

    async def _consume(self):
        connection_string = f'amqp://{self.spice.host_config["fingerprint"]}:{self.spice.host_config["rabbitmq_password"]}@{self.spice.host_config["rabbitmq_host"]}:{self.spice.host_config["rabbitmq_port"]}/agent'
        print(connection_string)
        connection = await aio_pika.connect_robust(
            connection_string,
            loop=self.loop,
        )

        async with connection:
            queue_name = "default"
            channel: aio_pika.abc.AbstractChannel = await connection.channel()
            queue: aio_pika.abc.AbstractQueue = await channel.get_queue(queue_name)

            task = None

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        print("message obtained ", message)
                        data = json.loads(message.body.decode("utf-8"))
                        status = data.get("status")
                        if status == "READY_FOR_PICKUP":
                            print("start")
                            task = asyncio.create_task(
                                self._claim_message_callback(message, data)
                            )
                            print("new task created", task)
                        elif status == "REQUEST_CANCEL":
                            if not task:
                                print("no task")
                                # await message.ack()
                            else:
                                print("request cancel")
                                # await message.ack()
                                inference_job_id = data.get("inference_job_id", None)
                                task.cancel()
                                # poll the status of the task
                                while True:
                                    # get the status of the task
                                    status = task.done()

                                    # report the status
                                    print(f">task done: {task.done()}")
                                    # check if the task is done
                                    if status:
                                        break
                                    # otherwise block for a moment
                                    await asyncio.sleep(0.1)

                                await self.spice.inference._async_update_inference_job(
                                    inference_job_id=inference_job_id,
                                    status="CANCELED",
                                )
                        else:
                            print("unhandled")
