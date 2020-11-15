#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Consumes activity entries from the RabbitMQ activity queue and writes to DB, printing out statistics as well
"""
import pika
import helpers
import time
import json
import database

count = 0

def init():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue=helpers.ACTIVITY_QUEUE, durable=True)

    # represents a lap of 1000 entries
    lap_time = time.time()
    start_time = time.time()

    # total number of usernames processed
    count = 0
    entry_count = 0
    lap_entries = 0

    def callback(ch, method, properties, body):
        nonlocal count, entry_count, lap_entries, lap_time, start_time

        # read payload and convert list of lists into list of tuples
        entries = json.loads(body.decode())
        entries = [tuple(e) for e in entries]

        if len(entries) > 0:
            username = entries[0][0]
            database.insert_activity_entries(entries)
            print("%d | Inserted %d activity entries for user %s" % (count, len(entries), username))

            count += 1
            entry_count += len(entries)
            lap_entries += len(entries)

            if (count % 100 == 0):
                t = time.time() - lap_time
                avg_users_per_sec = count/(time.time() - start_time)
                users_per_sec = 100 / t
                entries_per_sec = lap_entries / t
                print("Processed %d total users, %d total entries, %.3f | Rate of %.3f users/s, %.3f entries/s" % (
                count, entry_count, avg_users_per_sec, users_per_sec, entries_per_sec))
                lap_entries = 0
                lap_time = time.time()
        else:
            print("Empty body: ", body.decode())
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=helpers.ACTIVITY_QUEUE, on_message_callback=callback)
    channel.start_consuming()

if __name__ == "__main__":
    init()
