from ping3 import ping, verbose_ping

def ping_host(host):
    ping_result = verbose_ping(host, count=10)

    return ping_result.split()

hosts = [
    '129.21.22.239',
    '129.21.22.222'
]

for host in hosts:
    print(ping_host(host))