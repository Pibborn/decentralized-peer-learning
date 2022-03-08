# Decentralized Peer Learning

Multiple agents learning together.

## State of the code

The current code is able to perform peer learning by using the ```pfrl``` library. The main entry points are ```main.py``` and ```main_peer.py```.
There are also a bunch of ```.sh``` scripts which have been written to make the code run on MOGON. They might be ignored for now.

The ```dictator.py``` script represents a dictator implementation as it was intended by Aaron. I don't remember the full details, but it was not suitable because of some miscommunication issues.

I have started doing a rewrite of the code in ```run.py``` using ```stable-baselines3```. For now I have only implemented single-agent training :P 

