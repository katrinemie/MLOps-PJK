

Node: daki-master

Purpose: GPU Worker (Train/test/eval/serve)
IP-addr: (ikke angivet)
MAC-addr: (ikke angivet)
Hostname: daki-master
H/W:
	•	Intel Ultra 9 285k 24-cores
	•	64 GB DDR5
	•	2TB SSD
	•	NVIDIA RTX Pro 4000 Ada 20GB
S/W: Jenkins worker

⸻

Node: daki-gpu1

Purpose: GPU Worker (Train/test/eval/serve)
IP-addr: (ikke angivet)
MAC-addr: (ikke angivet)
Hostname: daki-gpu1
H/W:
	•	Intel Ultra 9 285k 24-cores
	•	64 GB DDR5
	•	2TB SSD
	•	NVIDIA RTX Pro 4000 Ada 20GB
S/W: Jenkins worker

⸻

Node: daki-gpu2

Purpose: GPU Worker (Train/test/eval/serve)
IP-addr: (ikke angivet)
MAC-addr: (ikke angivet)
Hostname: daki-gpu2
H/W:
	•	Intel Ultra 9 285k 24-cores
	•	64 GB DDR5
	•	2TB SSD
	•	NVIDIA RTX Pro 4000 Ada 20GB
S/W: Jenkins worker

⸻

Node: daki-storage

Purpose: Storage and CI/CD Orchestration
IP-addr: 172.24.198.42
MAC-addr: f4:f1:9e:42:be:18
Hostname: daki-storage
H/W:
	•	Intel Ultra 7 265 24-cores
	•	32 GB DDR5
	•	1TB SSD + 2x12TB HDD
S/W:
	•	MinIO server (S3): port 9001 *
	•	Jenkins master: port 8080
	•	Docker registry: port 5000
	•	MLFlow: port 5050

⸻

Node: daki-nano1

Purpose: Deployment
IP-addr: (ikke angivet)
MAC-addr: 3c:6d:66:14:ef:c5
Hostname: daki-nano1
H/W: NVIDIA Jetson Nano
S/W: Jetpack

⸻

Node: daki-nano2

Purpose: Deployment
IP-addr: (ikke angivet)
MAC-addr: 3c:6d:66:15:01:2b
Hostname: daki-nano2
H/W: NVIDIA Jetson Nano
S/W: Jetpack

⸻

Fælles login

Username and password for all nodes: daki/daki
* pwd: dakiminio (til MinIO)

