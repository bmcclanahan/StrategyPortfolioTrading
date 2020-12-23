from ibapi import wrapper
from ibapi import client
from ibapi import contract
from ibapi import order
import pandas as pd
import os.path as opath


class Wrapper(wrapper.EWrapper): # used to receive messsages from TWS
    def __init__(self):
        pass

class Client(client.EClient): # used to send messages to TWS
    def __init__(self, wrapper):
        client.EClient.__init__(self, wrapper)

class PositionsApp(Wrapper, Client):

    def __init__(self):
        Wrapper.__init__(self)
        Client.__init__(self, wrapper=self)
        self.positions = []
        self.positions_df = None
        self.__unq_id = None


    def position(self, account: str, contract: contract.Contract,
                 position: float, avg_cost: float):
        super().position(account, contract, position, avg_cost)
        self.positions.append(
            [contract.symbol, contract.secType, position, avg_cost,
             account]
        )
        print(
            "Position.", "Account:", account, "Symbol:", contract.symbol,
            "SecType:", contract.secType, "Currency:", contract.currency,
            "Position:", position, "Avg cost:", avg_cost
        )

    def positionEnd(self):
        super().positionEnd()
        self.positions_df = pd.DataFrame(
            self.positions,
            columns=['symbol', 'security_type', 'position', 'avg_cost',
                     'account']
        )
        self.cancelPositions()
        self.disconnect()

    def get_positions(self):
        unq_id = self.get_unique_id()
        req_id = f'positions_{unq_id}'
        self.reqPositions() # I think this function if from Client class

    def get_unique_id(self, filepath='counter.txt'):
        counter = 1
        if self.__unq_id is None:
            if not opath.exists(filepath):
                with open(filepath, 'w') as cnt_file:
                    cnt_file.write('1')
            else:
                with open(filepath, 'r') as cnt_file:
                    counter = int(cnt_file.read())
                with open(filepath, 'w') as cnt_file:
                    cnt_file.write(str(counter + 5))
        else:
            counter = self.__unq_id + 5
            self.__unq_id = counter
        return counter
