#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Pushes work to the RabbitMQ queue and coordinates the fetches, periodically printing rate information
"""
import pika
from collection import database, helpers


def init():
    # connect to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    # initialize queue
    channel.queue_delete(queue=helpers.USERNAME_QUEUE)
    channel.queue_declare(queue=helpers.USERNAME_QUEUE, durable=True)

    # get all users and sort them
    users = database.get_all_users()
    users = sorted(users, key=lambda user: user[0])
    completed_users = database.get_completed_users()

    for user in users:
        username = user[0]
        if username not in completed_users:
            # add to queue
            channel.basic_publish(exchange='',
                                  routing_key=helpers.USERNAME_QUEUE,
                                  body=username)

    print("Initialized queue with %d users to fetch" % get_queue_size(channel))
    connection.close()


# gets number of usernames in the queue
def get_queue_size(channel):
    return channel.queue_declare(queue=helpers.USERNAME_QUEUE, durable=True,
                                 exclusive=False,
                                 auto_delete=False).method.message_count


if __name__ == "__main__":
    init()
