using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Handles spawning and clearing of static obstacles inside a FoodCollector area.
/// Obstacles are instantiated from a prefab and placed randomly, respecting a minimum spacing
/// so they don’t overlap with agents, food, or each other.
/// </summary>

public class ObstacleSpawner : MonoBehaviour
{
    [Header("Prefab & Counts")]
    public GameObject obstaclePrefab;
    public int numObstacles = 8;

    [Header("Placement")]
    public float minSpacing = 1.5f;  // Minimum spacing from other objects (agents/food/obstacles)

    readonly List<Transform> spawned = new(); // Keep track of spawned obstacles


    /// <summary>
    /// Clears all spawned obstacles from the scene.
    /// Used before respawning a new set on reset.
    /// </summary>
    
    public void ClearAll()
    {
        for (int i = spawned.Count - 1; i >= 0; i--)
        {
            if (spawned[i]) Destroy(spawned[i].gameObject);
        }
        spawned.Clear();
    }

    /// <summary>
    /// Respawns a fresh set of obstacles within the given range around the center point.
    /// Ensures each obstacle is placed at least `minSpacing` units away from food, agents, walls, etc.
    /// </summary>
    /// <param name="center">The center position of the area (usually the FoodCollectorArea transform)</param>
    /// <param name="range">The half-width of the spawn range (obstacles are placed within ±range on X/Z)</param>
    
    public void RespawnObstacles(Vector3 center, float range)
    {
        ClearAll();

        int i = 0, guard = 0, maxTries = numObstacles * 50;
        while (i < numObstacles && guard++ < maxTries)
        {
            var pos = center + new Vector3(Random.Range(-range, range), 0.5f, Random.Range(-range, range));
            if (!IsFree(pos, minSpacing)) continue;

            var go = Instantiate(obstaclePrefab, pos, Quaternion.identity, transform);
            // Rigid, but static: BoxCollider + (optional) Rigidbody isKinematic=true
            if (!go.TryGetComponent<BoxCollider>(out _)) go.AddComponent<BoxCollider>();
            var rb = go.GetComponent<Rigidbody>();
            if (!rb) rb = go.AddComponent<Rigidbody>();
            rb.isKinematic = true; // static “rigid” obstacle (no physics cost)
            go.tag = "obstacle";

            spawned.Add(go.transform);
            i++;
        }
    }

    /// <summary>
    /// Checks whether a position is free of other objects within a given radius.
    /// Ignores the floor/court geometry but blocks agents, food, walls, and existing obstacles.
    /// </summary>

    static bool IsFree(Vector3 p, float radius)
    {
        var hits = Physics.OverlapSphere(p, radius, ~0, QueryTriggerInteraction.Ignore);
        foreach (var h in hits)
        {
            // Ignore the arena floor/court and other untagged static geometry
            var tag = h.tag;
            var name = h.transform.name;
            if (tag == "Untagged" || name.Contains("Floor") || name.Contains("Court"))
                continue;

            // Block if we’d overlap a relevant object
            if (tag == "agent" || tag == "frozenAgent" || tag == "food" || tag == "badFood" || tag == "obstacle" || tag == "wall")
                return false;
        }
        return true;
    }

    void Start()

    {
        RespawnObstacles(transform.position, 20f); // temporary visual test
    }

}
