import asyncio
import json
import logging
import os
import ssl
import sys

import pika
from pika.exceptions import AMQPConnectionError, ConnectionClosedByBroker
from retry import retry

from spice_agent.client import Spice
from spice_agent.utils.version import get_current_version
from spice_agent.worker.actions_2 import AsyncWorker


# define a coroutine for a task
async def task_coroutine():
    # report a message
    print("executing the task")
    # block for a moment
    await asyncio.sleep(1)


async def main(loop):
    spice = Spice(host="localhost:8000")
    async_worker = AsyncWorker(spice=spice, loop=loop)

    try:
        async with asyncio.TaskGroup() as task_group:
            task_1 = task_group.create_task(async_worker._consume())
    except KeyboardInterrupt:
        print("yay")
    # # report a message
    # print("main coroutine started")
    # # create and schedule the task
    # task = asyncio.create_task(task_coroutine())
    # # wait a moment
    # await asyncio.sleep(0.1)
    # # cancel the task
    # was_cancelled = task.cancel()
    # # report whether the cancel request was successful
    # print(f"was canceled: {was_cancelled}")
    # # wait a moment
    # await asyncio.sleep(0.1)
    # # check the status of the task
    # print(f"canceled: {task.cancelled()}")
    # # report a final message
    # print("main coroutine done")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
    loop.close()
