#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Retrieves work from the RabbitMQ queue and performs the fetch operation, then passes the resulting rows to the output queue
"""
import pika
from collection import helpers
import sys
from collection.retriever import Retriever
import json

def init():
    instance = int(sys.argv[1])
    retriever = Retriever(instance)
    print("Initialized Retriever connections for instance %d" % instance)

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue=helpers.USERNAME_QUEUE, durable=True)
    channel.queue_declare(queue=helpers.ACTIVITY_QUEUE, durable=True)

    def callback(ch, method, properties, body):
        username = body.decode()
        print("Received username %s" % username)

        # perform the fetch
        try:
            entries = retriever.get_activity_history(username)
            # write history to DB
            print("Fetched %d entries for user %s" % (len(entries), username))

            if(len(entries) > 0):
                channel.basic_publish(exchange='',
                                      routing_key=helpers.ACTIVITY_QUEUE,
                                      body=json.dumps(entries))
        except Exception as e:
            print("Failed to fetch for user %s : %s" % (username, str(e)))
        finally:
            # ack regardless of success
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=helpers.USERNAME_QUEUE, on_message_callback=callback)
    channel.start_consuming()

if __name__ == "__main__":
    init()
