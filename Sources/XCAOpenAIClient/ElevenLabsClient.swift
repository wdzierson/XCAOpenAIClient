import Foundation

public struct ElevenLabsClient {
    private let apiKey: String
    private let urlSession = URLSession.shared

    public init(apiKey: String) {
        self.apiKey = apiKey
    }

    public func generateSpeechFrom(text: String, voiceId: String, completion: @escaping (Result<Data, Error>) -> Void) {
        guard let url = URL(string: "https://api.elevenlabs.io/v1/text-to-speech/\(voiceId)") else {
            completion(.failure("Invalid URL"))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "xi-api-key")

        let jsonPayload: [String: Any] = [
            "text": text,
            "voice_settings": [
                "model_id": "eleven_multilingual_v2",
                "stability": 0.75,
                "similarity_boost": 0.75
            ]
        ]

        guard let httpBody = try? JSONSerialization.data(withJSONObject: jsonPayload, options: []) else {
            completion(.failure("Failed to serialize JSON"))
            return
        }

        request.httpBody = httpBody

        let task = urlSession.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }

            guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                completion(.failure("Failed with status code: \((response as? HTTPURLResponse)?.statusCode ?? -1)"))
                return
            }

            guard let data = data else {
                completion(.failure("No data received"))
                return
            }

            completion(.success(data))
        }

        task.resume()
    }
}
