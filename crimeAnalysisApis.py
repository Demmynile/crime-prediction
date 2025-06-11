import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.sql import text
from flask import Flask, jsonify, request

# Initialize the Flask app
app = Flask(__name__)

# CSV and DB config
csv_file_path = './exports/crime_df.csv'
db_url = 'postgresql+psycopg2://postgres:Harbeedeymee_123@localhost:5432/crime_db'
table_name = 'crime_db'

# Connect to PostgreSQL
engine = create_engine(db_url)
inspector = inspect(engine)

# Load CSV and insert data if table doesn't exist or is empty
def load_data():
    df = pd.read_csv(csv_file_path)
    with engine.connect() as conn:
        if table_name in inspector.get_table_names():
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            if result.scalar() == 0:
                df.to_sql(table_name, engine, index=False, if_exists='append')
                print("✅ Table exists but was empty — data inserted.")
            else:
                print("ℹ️ Table already has data — no action taken.")
        else:
            df.to_sql(table_name, engine, index=False, if_exists='replace')
            print("✅ Table created and data inserted.")

# Run data load once
load_data()

# === API Endpoints ===

# Root endpoint
@app.route('/')
def home():
    return jsonify({"message": "Crime DB API is running"})


# Get all records
@app.route('/api/crimes', methods=['GET'])
def get_crime_count():
    query = f"SELECT COUNT(*) as total FROM {table_name}"
    result = pd.read_sql_query(query, engine)
    return str(result['total'][0])


# Search crimes by column value (e.g. /api/search?column=city&value=Chicago)
@app.route('/api/geosearch', methods=['GET'])
def geosearch():
    lon = request.args.get('lon', type=float)
    lat = request.args.get('lat', type=float)
    radius = request.args.get('radius', default=1000, type=float)  # in meters

    if lon is None or lat is None:
        return jsonify({"error": "Missing 'lon' or 'lat' query parameters"}), 400

    try:
        query = text(f"""
            SELECT *,
                ST_AsText(ST_SetSRID(ST_MakePoint("Longitude", "Latitude"), 4326)) AS geom_wkt
            FROM crime_db
            WHERE ST_DWithin(
                ST_SetSRID(ST_MakePoint("Longitude", "Latitude"), 4326)::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                :radius
            )
            LIMIT 100
        """)
        df = pd.read_sql_query(query, engine, params={"lon": lon, "lat": lat, "radius": radius})
        return df.to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/monthly_crime', methods=['GET'])
def monthly_crime():
    try:
        query = text("""
            SELECT 
                TO_CHAR("Month"::date, 'FMMonth') AS month_name,
                COUNT(*) AS total_crimes
            FROM crime_db
            GROUP BY TO_CHAR("Month"::date, 'FMMonth')
            ORDER BY TO_DATE(TO_CHAR("Month"::date, 'FMMonth'), 'Month');
        """)

        df = pd.read_sql_query(query, engine)
        return df.to_json(orient='records')
    

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/crime_lsoa', methods=['GET'])
def crime_lsoa():
    try:
        query = text("""
            SELECT lsoa_code, COUNT(*) AS total_crimes
            FROM crime_db
            GROUP BY lsoa_code
            ORDER BY total_crimes DESC
            LIMIT 10;
        """)

        df = pd.read_sql_query(query, engine)
        return df.to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)}), 500  
    
@app.route('/api/crime_age', methods=['GET'])
def crime_age():
    try:
        query = text("""
            SELECT age, COUNT(*) AS total_crimes
            FROM crime_db
            GROUP BY age
            ORDER BY age;
        """)

        df = pd.read_sql_query(query, engine)
        return df.to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/crime_countries', methods=['GET'])
def crime_countries():
    try:
        query = text("""
            SELECT 
                "Country_of_Birth", 
                COUNT(*) AS total_crimes
            FROM crime_db
            GROUP BY "Country_of_Birth"
            ORDER BY total_crimes DESC;

        """)

        df = pd.read_sql_query(query, engine)
        return df.to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/crime_health', methods=['GET'])
def crime_health():
    try:
        query = text("""
            SELECT health, COUNT(*) AS total_crimes
            FROM crime_db
            GROUP BY health
            ORDER BY total_crimes DESC;
        """)

        df = pd.read_sql_query(query, engine)
        return df.to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/crime_occupation', methods=['GET'])
def get_crime_occupation():
    try:
        query = text("""
            SELECT "Occupation (former) (11 categories)", COUNT(*) AS total_crimes
            FROM crime_db
            GROUP BY "Occupation (former) (11 categories)"
            ORDER BY total_crimes DESC;

        """)
        df = pd.read_sql_query(query, engine)
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
