from typing import Callable, Optional

from torchvision.datasets import StanfordCars
from torchvision.datasets.vision import VisionDataset

__all__ = ['CARS196']
_cars_name = """
AM General Hummer SUV 2000
Acura RL Sedan 2012
Acura TL Sedan 2012
Acura TL Type-S 2008
Acura TSX Sedan 2012
Acura Integra Type R 2001
Acura ZDX Hatchback 2012
Aston Martin V8 Vantage Convertible 2012
Aston Martin V8 Vantage Coupe 2012
Aston Martin Virage Convertible 2012
Aston Martin Virage Coupe 2012
Audi RS 4 Convertible 2008
Audi A5 Coupe 2012
Audi TTS Coupe 2012
Audi R8 Coupe 2012
Audi V8 Sedan 1994
Audi 100 Sedan 1994
Audi 100 Wagon 1994
Audi TT Hatchback 2011
Audi S6 Sedan 2011
Audi S5 Convertible 2012
Audi S5 Coupe 2012
Audi S4 Sedan 2012
Audi S4 Sedan 2007
Audi TT RS Coupe 2012
BMW ActiveHybrid 5 Sedan 2012
BMW 1 Series Convertible 2012
BMW 1 Series Coupe 2012
BMW 3 Series Sedan 2012
BMW 3 Series Wagon 2012
BMW 6 Series Convertible 2007
BMW X5 SUV 2007
BMW X6 SUV 2012
BMW M3 Coupe 2012
BMW M5 Sedan 2010
BMW M6 Convertible 2010
BMW X3 SUV 2012
BMW Z4 Convertible 2012
Bentley Continental Supersports Conv. Convertible 2012
Bentley Arnage Sedan 2009
Bentley Mulsanne Sedan 2011
Bentley Continental GT Coupe 2012
Bentley Continental GT Coupe 2007
Bentley Continental Flying Spur Sedan 2007
Bugatti Veyron 16.4 Convertible 2009
Bugatti Veyron 16.4 Coupe 2009
Buick Regal GS 2012
Buick Rainier SUV 2007
Buick Verano Sedan 2012
Buick Enclave SUV 2012
Cadillac CTS-V Sedan 2012
Cadillac SRX SUV 2012
Cadillac Escalade EXT Crew Cab 2007
Chevrolet Silverado 1500 Hybrid Crew Cab 2012
Chevrolet Corvette Convertible 2012
Chevrolet Corvette ZR1 2012
Chevrolet Corvette Ron Fellows Edition Z06 2007
Chevrolet Traverse SUV 2012
Chevrolet Camaro Convertible 2012
Chevrolet HHR SS 2010
Chevrolet Impala Sedan 2007
Chevrolet Tahoe Hybrid SUV 2012
Chevrolet Sonic Sedan 2012
Chevrolet Express Cargo Van 2007
Chevrolet Avalanche Crew Cab 2012
Chevrolet Cobalt SS 2010
Chevrolet Malibu Hybrid Sedan 2010
Chevrolet TrailBlazer SS 2009
Chevrolet Silverado 2500HD Regular Cab 2012
Chevrolet Silverado 1500 Classic Extended Cab 2007
Chevrolet Express Van 2007
Chevrolet Monte Carlo Coupe 2007
Chevrolet Malibu Sedan 2007
Chevrolet Silverado 1500 Extended Cab 2012
Chevrolet Silverado 1500 Regular Cab 2012
Chrysler Aspen SUV 2009
Chrysler Sebring Convertible 2010
Chrysler Town and Country Minivan 2012
Chrysler 300 SRT-8 2010
Chrysler Crossfire Convertible 2008
Chrysler PT Cruiser Convertible 2008
Daewoo Nubira Wagon 2002
Dodge Caliber Wagon 2012
Dodge Caliber Wagon 2007
Dodge Caravan Minivan 1997
Dodge Ram Pickup 3500 Crew Cab 2010
Dodge Ram Pickup 3500 Quad Cab 2009
Dodge Sprinter Cargo Van 2009
Dodge Journey SUV 2012
Dodge Dakota Crew Cab 2010
Dodge Dakota Club Cab 2007
Dodge Magnum Wagon 2008
Dodge Challenger SRT8 2011
Dodge Durango SUV 2012
Dodge Durango SUV 2007
Dodge Charger Sedan 2012
Dodge Charger SRT-8 2009
Eagle Talon Hatchback 1998
FIAT 500 Abarth 2012
FIAT 500 Convertible 2012
Ferrari FF Coupe 2012
Ferrari California Convertible 2012
Ferrari 458 Italia Convertible 2012
Ferrari 458 Italia Coupe 2012
Fisker Karma Sedan 2012
Ford F-450 Super Duty Crew Cab 2012
Ford Mustang Convertible 2007
Ford Freestar Minivan 2007
Ford Expedition EL SUV 2009
Ford Edge SUV 2012
Ford Ranger SuperCab 2011
Ford GT Coupe 2006
Ford F-150 Regular Cab 2012
Ford F-150 Regular Cab 2007
Ford Focus Sedan 2007
Ford E-Series Wagon Van 2012
Ford Fiesta Sedan 2012
GMC Terrain SUV 2012
GMC Savana Van 2012
GMC Yukon Hybrid SUV 2012
GMC Acadia SUV 2012
GMC Canyon Extended Cab 2012
Geo Metro Convertible 1993
HUMMER H3T Crew Cab 2010
HUMMER H2 SUT Crew Cab 2009
Honda Odyssey Minivan 2012
Honda Odyssey Minivan 2007
Honda Accord Coupe 2012
Honda Accord Sedan 2012
Hyundai Veloster Hatchback 2012
Hyundai Santa Fe SUV 2012
Hyundai Tucson SUV 2012
Hyundai Veracruz SUV 2012
Hyundai Sonata Hybrid Sedan 2012
Hyundai Elantra Sedan 2007
Hyundai Accent Sedan 2012
Hyundai Genesis Sedan 2012
Hyundai Sonata Sedan 2012
Hyundai Elantra Touring Hatchback 2012
Hyundai Azera Sedan 2012
Infiniti G Coupe IPL 2012
Infiniti QX56 SUV 2011
Isuzu Ascender SUV 2008
Jaguar XK XKR 2012
Jeep Patriot SUV 2012
Jeep Wrangler SUV 2012
Jeep Liberty SUV 2012
Jeep Grand Cherokee SUV 2012
Jeep Compass SUV 2012
Lamborghini Reventon Coupe 2008
Lamborghini Aventador Coupe 2012
Lamborghini Gallardo LP 570-4 Superleggera 2012
Lamborghini Diablo Coupe 2001
Land Rover Range Rover SUV 2012
Land Rover LR2 SUV 2012
Lincoln Town Car Sedan 2011
MINI Cooper Roadster Convertible 2012
Maybach Landaulet Convertible 2012
Mazda Tribute SUV 2011
McLaren MP4-12C Coupe 2012
Mercedes-Benz 300-Class Convertible 1993
Mercedes-Benz C-Class Sedan 2012
Mercedes-Benz SL-Class Coupe 2009
Mercedes-Benz E-Class Sedan 2012
Mercedes-Benz S-Class Sedan 2012
Mercedes-Benz Sprinter Van 2012
Mitsubishi Lancer Sedan 2012
Nissan Leaf Hatchback 2012
Nissan NV Passenger Van 2012
Nissan Juke Hatchback 2012
Nissan 240SX Coupe 1998
Plymouth Neon Coupe 1999
Porsche Panamera Sedan 2012
Ram C/V Cargo Van Minivan 2012
Rolls-Royce Phantom Drophead Coupe Convertible 2012
Rolls-Royce Ghost Sedan 2012
Rolls-Royce Phantom Sedan 2012
Scion xD Hatchback 2012
Spyker C8 Convertible 2009
Spyker C8 Coupe 2009
Suzuki Aerio Sedan 2007
Suzuki Kizashi Sedan 2012
Suzuki SX4 Hatchback 2012
Suzuki SX4 Sedan 2012
Tesla Model S Sedan 2012
Toyota Sequoia SUV 2012
Toyota Camry Sedan 2012
Toyota Corolla Sedan 2012
Toyota 4Runner SUV 2012
Volkswagen Golf Hatchback 2012
Volkswagen Golf Hatchback 1991
Volkswagen Beetle Hatchback 2012
Volvo C30 Hatchback 2012
Volvo 240 Sedan 1993
Volvo XC90 SUV 2007
smart fortwo Convertible 2012
"""
_cars_mapping = {52: 0,
                 53: 0,
                 64: 0,
                 68: 0,
                 69: 0,
                 73: 0,
                 74: 0,
                 85: 0,
                 86: 0,
                 89: 0,
                 90: 0,
                 105: 0,
                 110: 0,
                 112: 0,
                 113: 0,
                 121: 0,
                 123: 0,
                 124: 0,
                 1: 1,
                 2: 1,
                 3: 1,
                 4: 1,
                 15: 1,
                 16: 1,
                 19: 1,
                 22: 1,
                 23: 1,
                 25: 1,
                 28: 1,
                 34: 1,
                 39: 1,
                 40: 1,
                 43: 1,
                 46: 1,
                 48: 1,
                 50: 1,
                 60: 1,
                 62: 1,
                 66: 1,
                 72: 1,
                 78: 1,
                 95: 1,
                 96: 1,
                 104: 1,
                 114: 1,
                 116: 1,
                 128: 1,
                 133: 1,
                 134: 1,
                 135: 1,
                 136: 1,
                 137: 1,
                 139: 1,
                 155: 1,
                 161: 1,
                 163: 1,
                 164: 1,
                 166: 1,
                 172: 1,
                 175: 1,
                 176: 1,
                 180: 1,
                 181: 1,
                 183: 1,
                 184: 1,
                 186: 1,
                 187: 1,
                 193: 1,
                 0: 2,
                 31: 2,
                 32: 2,
                 36: 2,
                 47: 2,
                 49: 2,
                 51: 2,
                 57: 2,
                 59: 2,
                 61: 2,
                 67: 2,
                 75: 2,
                 88: 2,
                 93: 2,
                 94: 2,
                 108: 2,
                 109: 2,
                 117: 2,
                 119: 2,
                 120: 2,
                 130: 2,
                 131: 2,
                 132: 2,
                 141: 2,
                 142: 2,
                 144: 2,
                 145: 2,
                 146: 2,
                 147: 2,
                 148: 2,
                 153: 2,
                 154: 2,
                 158: 2,
                 185: 2,
                 188: 2,
                 194: 2,
                 7: 3,
                 9: 3,
                 11: 3,
                 20: 3,
                 26: 3,
                 30: 3,
                 35: 3,
                 37: 3,
                 38: 3,
                 44: 3,
                 54: 3,
                 55: 3,
                 56: 3,
                 58: 3,
                 76: 3,
                 79: 3,
                 80: 3,
                 99: 3,
                 101: 3,
                 102: 3,
                 106: 3,
                 122: 3,
                 143: 3,
                 151: 3,
                 156: 3,
                 157: 3,
                 160: 3,
                 174: 3,
                 178: 3,
                 195: 3,
                 5: 4,
                 8: 4,
                 10: 4,
                 12: 4,
                 13: 4,
                 14: 4,
                 21: 4,
                 24: 4,
                 27: 4,
                 33: 4,
                 41: 4,
                 42: 4,
                 45: 4,
                 65: 4,
                 71: 4,
                 92: 4,
                 100: 4,
                 103: 4,
                 111: 4,
                 127: 4,
                 140: 4,
                 149: 4,
                 150: 4,
                 152: 4,
                 159: 4,
                 162: 4,
                 170: 4,
                 171: 4,
                 179: 4,
                 6: 5,
                 18: 5,
                 97: 5,
                 98: 5,
                 129: 5,
                 138: 5,
                 167: 5,
                 169: 5,
                 177: 5,
                 182: 5,
                 189: 5,
                 190: 5,
                 191: 5,
                 192: 5,
                 17: 6,
                 29: 6,
                 81: 6,
                 82: 6,
                 83: 6,
                 91: 6,
                 63: 7,
                 70: 7,
                 77: 7,
                 84: 7,
                 87: 7,
                 107: 7,
                 115: 7,
                 118: 7,
                 125: 7,
                 126: 7,
                 165: 7,
                 168: 7,
                 173: 7
                 }


class CARS196(VisionDataset):
    """
    The coarse labels are generated by the mapping file: '_cars_mapping' dict.
    """
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None):
        super().__init__(root, transform=transform)
        self.dataset = StanfordCars(root=root, split=split, transform=transform, download=True)
        self.coarse_mapping = _cars_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, fine_label = self.dataset[index]
        coarse_label = self.coarse_mapping[fine_label]
        return image, coarse_label, fine_label
