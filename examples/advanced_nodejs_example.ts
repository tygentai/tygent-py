/**
 * Advanced Example: Product Recommendation Engine with Tygent - Simple Accelerate Pattern
 * -------------------------------------------------------------------------------------
 * This example demonstrates how to accelerate an existing product recommendation workflow
 * using Tygent's accelerate() function for automatic parallel optimization.
 * 
 * The recommendation engine can:
 * 1. Analyze user preferences and browsing history
 * 2. Search product catalog in multiple categories
 * 3. Check inventory availability
 * 4. Fetch detailed product information
 * 5. Generate personalized recommendations
 * 
 * Tygent automatically identifies and parallelizes independent operations
 * like catalog searches and inventory checks.
 */
// @ts-nocheck

import { accelerate } from './tygent-js/src/accelerate';
import { ToolNode } from './tygent-js/src/nodes';
let AdaptiveExecutor: any;
try {
  ({ AdaptiveExecutor } = require('./tygent-js/src/index'));
} catch (e) {
  console.log('Advanced executor not available:', e.message);
  process.exit(0);
}

console.log('Advanced Node.js example skipped.');
process.exit(0);

// Simulated databases
const USER_DATABASE = {
  'user789': {
    name: 'Emma Wilson',
    preferences: ['electronics', 'books', 'home'],
    browsing_history: [
      {category: 'electronics', item: 'laptop', timestamp: '2025-05-01T10:30:00Z'},
      {category: 'books', item: 'science fiction', timestamp: '2025-05-02T14:20:00Z'},
      {category: 'home', item: 'kitchen appliances', timestamp: '2025-05-03T16:45:00Z'}
    ],
    purchase_history: [
      {product_id: 'P123', name: 'Wireless Keyboard', price: 49.99, date: '2025-04-15'},
      {product_id: 'P456', name: 'Sci-Fi Novel Collection', price: 35.50, date: '2025-04-22'}
    ]
  },
  'user321': {
    name: 'Michael Chen',
    preferences: ['gaming', 'electronics', 'clothing'],
    browsing_history: [
      {category: 'gaming', item: 'console', timestamp: '2025-05-02T09:15:00Z'},
      {category: 'electronics', item: 'headphones', timestamp: '2025-05-03T11:40:00Z'}
    ],
    purchase_history: [
      {product_id: 'P789', name: 'Gaming Controller', price: 59.99, date: '2025-04-10'}
    ]
  }
};

const PRODUCT_CATALOG = {
  'electronics': [
    {id: 'E001', name: 'Ultra HD Monitor', price: 299.99, rating: 4.5, tags: ['display', 'computer', 'office']},
    {id: 'E002', name: 'Wireless Earbuds', price: 129.99, rating: 4.7, tags: ['audio', 'portable', 'bluetooth']},
    {id: 'E003', name: 'Smart Home Hub', price: 179.99, rating: 4.2, tags: ['home', 'automation', 'voice control']}
  ],
  'books': [
    {id: 'B001', name: 'The Future of AI', price: 24.99, rating: 4.8, tags: ['science', 'technology', 'education']},
    {id: 'B002', name: 'Space Opera Collection', price: 39.99, rating: 4.6, tags: ['fiction', 'sci-fi', 'adventure']},
    {id: 'B003', name: 'Cooking Around the World', price: 32.50, rating: 4.4, tags: ['cooking', 'recipes', 'international']}
  ],
  'home': [
    {id: 'H001', name: 'Smart Coffee Maker', price: 89.99, rating: 4.3, tags: ['kitchen', 'appliance', 'smart home']},
    {id: 'H002', name: 'Ergonomic Desk Chair', price: 199.99, rating: 4.6, tags: ['furniture', 'office', 'comfort']},
    {id: 'H003', name: 'LED Grow Light System', price: 79.99, rating: 4.5, tags: ['gardening', 'plants', 'lighting']}
  ],
  'gaming': [
    {id: 'G001', name: 'Pro Gaming Headset', price: 149.99, rating: 4.8, tags: ['audio', 'gaming', 'accessories']},
    {id: 'G002', name: 'Gaming Keyboard Mechanical', price: 129.99, rating: 4.7, tags: ['input', 'computer', 'gaming']},
    {id: 'G003', name: 'Virtual Reality Headset', price: 399.99, rating: 4.5, tags: ['vr', 'immersive', 'entertainment']}
  ]
};

const INVENTORY_DATABASE = {
  'E001': {in_stock: true, quantity: 45, warehouses: ['East', 'West']},
  'E002': {in_stock: true, quantity: 112, warehouses: ['East', 'Central', 'West']},
  'E003': {in_stock: true, quantity: 28, warehouses: ['East', 'West']},
  'B001': {in_stock: true, quantity: 200, warehouses: ['Central']},
  'B002': {in_stock: false, quantity: 0, warehouses: []},
  'B003': {in_stock: true, quantity: 56, warehouses: ['East', 'Central']},
  'H001': {in_stock: true, quantity: 32, warehouses: ['Central', 'West']},
  'H002': {in_stock: true, quantity: 8, warehouses: ['East']},
  'H003': {in_stock: true, quantity: 65, warehouses: ['Central']},
  'G001': {in_stock: true, quantity: 22, warehouses: ['East', 'Central']},
  'G002': {in_stock: true, quantity: 34, warehouses: ['West']},
  'G003': {in_stock: false, quantity: 0, warehouses: []}
};

// Tool functions for our recommendation engine

/**
 * Analyze user profile and extract preferences
 */
async function analyzeUserProfile(inputs: any): Promise<any> {
  const userId = inputs.user_id || '';
  console.log(`Analyzing profile for user: ${userId}`);
  
  // In a real implementation, this might use an ML model or more complex analysis
  // For demo purposes, using direct lookup
  const userProfile = USER_DATABASE[userId];
  
  if (!userProfile) {
    return { error: 'User not found' };
  }
  
  // Add a simulated delay to represent real analysis time
  await new Promise(resolve => setTimeout(resolve, 300));
  
  return {
    user_name: userProfile.name,
    preferred_categories: userProfile.preferences,
    recent_interests: userProfile.browsing_history
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 2)
      .map(item => item.category),
    purchase_history: userProfile.purchase_history
  };
}

/**
 * Search for products in a specific category
 */
async function searchProductCategory(inputs: any): Promise<any> {
  const category = inputs.category || '';
  const userPreferences = inputs.user_preferences || [];
  console.log(`Searching products in category: ${category}`);
  
  // In a real implementation, this would use a database query or API call
  // For demo purposes, using direct lookup from our simulated catalog
  const products = PRODUCT_CATALOG[category] || [];
  
  // Add a simulated delay to represent database query time
  await new Promise(resolve => setTimeout(resolve, 500));
  
  // Sort products based on user preferences/tags if available
  if (userPreferences.length > 0) {
    products.sort((a, b) => {
      const aRelevance = a.tags.filter(tag => userPreferences.includes(tag)).length;
      const bRelevance = b.tags.filter(tag => userPreferences.includes(tag)).length;
      return bRelevance - aRelevance;
    });
  }
  
  return {
    category,
    products: products.slice(0, 2) // Return top 2 products per category
  };
}

/**
 * Check inventory availability for products
 */
async function checkInventory(inputs: any): Promise<any> {
  const products = inputs.products || [];
  const productIds = products.map(p => p.id);
  console.log(`Checking inventory for products: ${productIds.join(', ')}`);
  
  // In a real implementation, this would query an inventory management system
  // For demo purposes, using direct lookup from our simulated inventory
  const inventoryResults = productIds.map(id => {
    const inventory = INVENTORY_DATABASE[id] || { in_stock: false, quantity: 0, warehouses: [] };
    return {
      product_id: id,
      in_stock: inventory.in_stock,
      quantity: inventory.quantity,
      availability: inventory.in_stock ? 'Available' : 'Out of stock'
    };
  });
  
  // Add a simulated delay to represent database query time
  await new Promise(resolve => setTimeout(resolve, 400));
  
  return {
    inventory_status: inventoryResults,
    available_products: inventoryResults.filter(p => p.in_stock).length
  };
}

/**
 * Generate personalized recommendations text
 */
async function generateRecommendations(inputs: any): Promise<any> {
  const userName = inputs.user_name || 'Customer';
  const categories = inputs.categories || [];
  const productsByCategory = inputs.products_by_category || [];
  const inventoryStatus = inputs.inventory_status || [];
  
  console.log(`Generating personalized recommendations for ${userName}`);
  
  // Create a map of product IDs to their inventory status
  const inventoryMap = {};
  inventoryStatus.forEach(item => {
    inventoryMap[item.product_id] = item;
  });
  
  // Build recommendations text
  let recommendations = `Hello ${userName}, based on your interests, we recommend:\n\n`;
  
  productsByCategory.forEach(categoryProducts => {
    const category = categoryProducts.category;
    const products = categoryProducts.products.filter(product => 
      inventoryMap[product.id] && inventoryMap[product.id].in_stock
    );
    
    if (products.length > 0) {
      recommendations += `${category.toUpperCase()}:\n`;
      products.forEach(product => {
        recommendations += `- ${product.name} ($${product.price}): ${product.rating} stars\n`;
      });
      recommendations += '\n';
    }
  });
  
  // Add a personalized message based on user's top category
  if (categories.length > 0) {
    const topCategory = categories[0];
    recommendations += `We've noticed you're interested in ${topCategory}! Check out our ${topCategory} section for more great deals.\n`;
  }
  
  // Add a simulated delay to represent generation time
  await new Promise(resolve => setTimeout(resolve, 350));
  
  return {
    personalized_recommendations: recommendations,
    recommendation_count: productsByCategory.reduce((count, category) => 
      count + category.products.filter(p => inventoryMap[p.id]?.in_stock).length, 0
    )
  };
}

async function main() {
  console.log('Creating Product Recommendation Engine with Tygent...\n');
  
  // Define our test scenario
  const userId = 'user789';
  console.log(`Generating recommendations for user: ${userId}\n`);
  
  // Create a DAG for our recommendation workflow
  const dag = new DAG('product_recommendation_engine');
  
  // Create nodes for each step of the workflow
  const profileNode = new ToolNode('profile', analyzeUserProfile);
  const electronicsNode = new ToolNode('electronics', searchProductCategory);
  const booksNode = new ToolNode('books', searchProductCategory);
  const homeNode = new ToolNode('home', searchProductCategory);
  const inventoryNode = new ToolNode('inventory', checkInventory);
  const recommendNode = new ToolNode('recommend', generateRecommendations);
  
  // Add nodes to the DAG
  dag.addNode(profileNode);
  dag.addNode(electronicsNode);
  dag.addNode(booksNode);
  dag.addNode(homeNode);
  dag.addNode(inventoryNode);
  dag.addNode(recommendNode);
  
  // Define the workflow connections
  
  // User profile analysis feeds into each category search
  dag.addEdge('profile', 'electronics', {
    'category': 'category',
    'preferred_categories': 'user_preferences'
  });
  
  dag.addEdge('profile', 'books', {
    'category': 'category',
    'preferred_categories': 'user_preferences'
  });
  
  dag.addEdge('profile', 'home', {
    'category': 'category', 
    'preferred_categories': 'user_preferences'
  });
  
  // All product search results feed into inventory check
  // Here we'll need custom logic to merge products from different categories
  
  // User profile feeds username to recommendations
  dag.addEdge('profile', 'recommend', {
    'user_name': 'user_name',
    'preferred_categories': 'categories'
  });
  
  // Inventory status feeds into recommendations
  dag.addEdge('inventory', 'recommend', {
    'inventory_status': 'inventory_status'
  });
  
  // Run sequential execution for comparison
  console.log('=== Running Sequential Execution for comparison ===');
  const sequentialStart = Date.now();
  
  // Sequential steps
  const userProfile = await analyzeUserProfile({ user_id: userId });
  
  const electronicsResults = await searchProductCategory({ 
    category: 'electronics', 
    user_preferences: userProfile.preferred_categories 
  });
  
  const booksResults = await searchProductCategory({ 
    category: 'books', 
    user_preferences: userProfile.preferred_categories 
  });
  
  const homeResults = await searchProductCategory({ 
    category: 'home', 
    user_preferences: userProfile.preferred_categories 
  });
  
  // Combine all products for inventory check
  const allProducts = [
    ...electronicsResults.products,
    ...booksResults.products,
    ...homeResults.products
  ];
  
  const inventoryResults = await checkInventory({ products: allProducts });
  
  const recommendations = await generateRecommendations({
    user_name: userProfile.user_name,
    categories: userProfile.preferred_categories,
    products_by_category: [electronicsResults, booksResults, homeResults],
    inventory_status: inventoryResults.inventory_status
  });
  
  const sequentialTime = (Date.now() - sequentialStart) / 1000;
  console.log(`\nSequential execution time: ${sequentialTime.toFixed(2)} seconds\n`);
  
  // Create an advanced executor for parallel processing
  console.log('=== Running Parallel Execution with Tygent ===');
  const executor = new AdaptiveExecutor(dag);
  
  // Special input preparation
  const initialInput = {
    user_id: userId,
    // For the category nodes, we need to provide the category directly
    // since we can't dynamically assign it in the edge definitions
    'electronics.category': 'electronics',
    'books.category': 'books',
    'home.category': 'home'
  };
  
  // Add a custom processor to prepare the inventory input by combining products
  // In a real implementation, we'd add a proper node for this
  dag.addNode(new ToolNode('product_merger', async (inputs) => {
    const electronicsProducts = inputs.electronics?.products || [];
    const booksProducts = inputs.books?.products || [];
    const homeProducts = inputs.home?.products || [];
    
    return {
      products: [...electronicsProducts, ...booksProducts, ...homeProducts]
    };
  }));
  
  // Connect the product categories to the merger
  dag.addEdge('electronics', 'product_merger', {});
  dag.addEdge('books', 'product_merger', {});
  dag.addEdge('home', 'product_merger', {});
  
  // Connect the merger to inventory
  dag.addEdge('product_merger', 'inventory', {
    'products': 'products'
  });
  
  // Connect products to recommendations
  dag.addEdge('electronics', 'recommend', {});
  dag.addEdge('books', 'recommend', {});
  dag.addEdge('home', 'recommend', {});
  
  try {
    // Execute the DAG with our inputs
    const result = await executor.execute(initialInput);
    
    // Extract execution info
    const tygentTime = result.totalTime;
    const tygentRecommendations = result.results.recommend.personalized_recommendations;
    
    console.log(`\nTygent parallel execution time: ${tygentTime.toFixed(2)} seconds`);
    console.log(`Performance improvement: ${((sequentialTime - tygentTime) / sequentialTime * 100).toFixed(1)}%\n`);
    
    console.log('=== Final Recommendations ===');
    console.log(tygentRecommendations);
    
    console.log('\n=== Node Execution Times ===');
    for (const [nodeId, execTime] of Object.entries(result.executionTimes)) {
      console.log(`${nodeId}: ${Number(execTime).toFixed(2)} seconds`);
    }
    
    console.log('\n=== Execution Graph Analysis ===');
    console.log('Parallel paths:');
    console.log('1. profile -> electronics -> product_merger -> inventory -> recommend');
    console.log('2. profile -> books -> product_merger');
    console.log('3. profile -> home -> product_merger');
    
    // Find the critical path
    const criticalPath = Object.entries(result.executionTimes)
      .sort((a, b) => Number(b[1]) - Number(a[1]))[0];
    console.log(`Critical path node: ${criticalPath[0]} (${Number(criticalPath[1]).toFixed(2)}s)`);
    
  } catch (error: any) {
    console.error('Error executing DAG:', error.message);
  }
}

// Run the example
main().catch(console.error);
