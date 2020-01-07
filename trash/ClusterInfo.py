class ClusterInfo:
  def __init__(self, nHits, minWire, maxWire, avgWire,  minTime, maxTime, avgTime):
    self.nHits   = nHits
    self.minWire = minWire
    self.maxWire = maxWire
    self.avgWire = avgWire
    self.minTime = minTime
    self.maxTime = maxTime
    self.avgTime = avgTime

  def wireWidth(self):
    return self.maxWire - self.minWire

  def timeWidth(self):
    return self.maxTime - self.minTime

