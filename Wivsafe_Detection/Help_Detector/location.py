import geocoder

class LocationService:
    def __init__(self):
        self.g = None
    
    def get_current_location_details(self):
        try:
            self.g = geocoder.ip('me')  # Get location information using IP
            if self.g.latlng is not None:
                location_details = {
                    'coordinates': self.g.latlng,
                    'city': self.g.city,
                    'state': self.g.state,
                    'country': self.g.country,
                    'address': self.g.address
                }
                return location_details
            else:
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def get_location_and_display(self):
        location_details = self.get_current_location_details()
        
        if location_details is not None:
            latitude, longitude = location_details['coordinates']
            print(f"Your current GPS coordinates are:")
            print(f"Latitude: {latitude}")
            print(f"Longitude: {longitude}")
            
            print(f"City: {location_details['city']}")
            print(f"State: {location_details['state']}")
            print(f"Country: {location_details['country']}")
            print(f"Address: {location_details['address']}")
        else:
            print("Unable to retrieve your location details.")
