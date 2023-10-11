//
//  Fitness_VisionApp.swift
//  Fitness Vision
//
//  Created by Emma Fu on 2023-09-18.
//

import SwiftUI

@main
struct Fitness_VisionApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
//                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
